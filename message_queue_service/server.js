const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");
const cors = require("cors");
const { StringDecoder } = require('string_decoder');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" },
});

require('dotenv').config();

app.use(express.static("public"));
app.use(cors());

const queue = []; // Hàng đợi lưu câu hỏi
const activeClients = {}; // Lưu danh sách client trong group
const chatHistories = {}; // Lưu lịch sử chat theo socket.id
const MAX_HISTORY_LENGTH = 10; // Tối đa 10 item hỏi/đáp

function addHistory(id, role, msg) {
    if (!msg) return;
    if (!chatHistories[id]) {
      chatHistories[id] = { sessionId: null, conversation: [] };
    }
    chatHistories[id].conversation.push({ role, content: msg });
    if (chatHistories[id].conversation.length > MAX_HISTORY_LENGTH) {
      chatHistories[id].conversation.shift();
    }
  }

io.on("connection", (socket) => {
    console.log("User connected:", socket.id);

    // Khi user join vào group chat
    socket.on("joinGroup", (groupId) => {
        socket.join(socket.id);
        activeClients[socket.id] = socket.id;
        chatHistories[socket.id] = { sessionId: null, conversation: [] };
        console.log(`User ${socket.id} joined group ${groupId}`);
    });

    // Khi user gửi câu hỏi
    socket.on("sendQuestion", ({ question }) => {
        if (!question || !question.trim()) return;
        queue.push({ clientId: socket.id, question: question.trim() });
        addHistory(socket.id, "user", question.trim());
    });

    socket.on("disconnect", () => {
        console.log("User disconnected:", socket.id);
        delete activeClients[socket.id];
        delete chatHistories[socket.id];

        // Xóa tất cả câu hỏi của client đã disconnect khỏi hàng đợi
        const newQueue = queue.filter((item) => item.clientId !== socket.id);
        queue.length = 0; // Xóa hết phần tử cũ
        queue.push(...newQueue); // Gán lại hàng đợi mới

        console.log("Updated queue after disconnect:", queue);
    });
});

let runningCount = 0;
function getApiStreamTimeout() {
    return process.env.API_STREAM_TIMEOUT_MS ? parseInt(process.env.API_STREAM_TIMEOUT_MS) : 30000;
}

// Gọi API và gửi từng phần response về đúng group
async function callStreamingAPI(groupId, question) {
    try {
        console.log(`Bắt đầu call ${groupId}: ${question}`);
        runningCount++;
        const decoder = new StringDecoder('utf8');
        const headers = {};
        const hist = chatHistories[groupId];
        if (hist && hist.sessionId) {
            headers["X-Session-ID"] = hist.sessionId;
        }

        const response = await axios({
            method: "POST", // Hoặc "POST" nếu API yêu cầu
            url: process.env.AGENT_SERVICE_URL,
            headers,
            data: { 
                "question": question,
                "userid": groupId,
                "conversation": hist ? hist.conversation : []
             }, // Gửi câu hỏi theo query string (nếu API hỗ trợ)
            responseType: "stream",
        });
        const newSessionId = response.headers["x-session-id"];
        if (newSessionId) {
            if (!chatHistories[groupId]) {
                chatHistories[groupId] = { sessionId: newSessionId, conversation: [] };
            } else {
                chatHistories[groupId].sessionId = newSessionId;
            }
        }
        
        let fullResponse = ""; // Gom dữ liệu trả về
        let receivedFirstChunk = false;
        let apiStreamTimeout = getApiStreamTimeout();
        const watchdog = setTimeout(() => {
            if (!receivedFirstChunk) {
                response.data.destroy();
                io.to(groupId).emit("response", { text: `Timeout: Không nhận được phản hồi từ API sau ${apiStreamTimeout} mili giây.`, isEnd: true });
                console.warn(`[${groupId}] Request bị huỷ do timeout.`);
                runningCount--;
            }
        }, apiStreamTimeout);

        response.data.on("data", (chunk) => {
            const text = decoder.write(chunk);
            if (text) {
                if (!receivedFirstChunk) {
                    receivedFirstChunk = true;
                    clearTimeout(watchdog);
                }
                fullResponse += text;
                io.to(groupId).emit("response", { text });
            }
        });

        response.data.on("end", () => {
            clearTimeout(watchdog);
            io.to(groupId).emit("response", { text:"", isEnd:true });
            response.data.removeAllListeners(); // Giải phóng bộ nhớ tránh memory leak
            addHistory(groupId, "assistant", fullResponse);
            runningCount--;
            const timestamp = new Date().toISOString();
            const sessionId = chatHistories[groupId]?.sessionId || 'N/A';
            console.log(
                `[${timestamp}] [END] sessionId=${sessionId} ` +
                `groupId=${groupId} question="${question}" ` +
                `fullResponse="${fullResponse.replace(/\n/g, '\\n')}"`
            );
        });

        response.data.on("error", (err) => {
            clearTimeout(watchdog);
            console.error("Lỗi khi nhận dữ liệu từ API:", err);
            io.to(groupId).emit("response", { text: "Lỗi từ API", isEnd:true });
            runningCount--;
        });
    } catch (error) {
        console.error("Lỗi khi gọi API:", error.message);
        io.to(groupId).emit("response", { text: "Lỗi kết nối API", isEnd:true });
        runningCount--;
    }
}

function getMaxConcurentCalls() {
    return process.env.MAX_CONCURRENT_CALLS ? parseInt(process.env.MAX_CONCURRENT_CALLS) : 1;
}

async function processQueue() {
    while (true) {
        if (queue.length > 0 && runningCount < getMaxConcurentCalls()) {
            const item = queue.shift();
            if (item) {
                const { clientId, question } = item;
                try {
                    callStreamingAPI(clientId, question);
                } catch (err) {
                    console.error(err);
                }
            }
        } else {
            await new Promise((resolve) => setTimeout(resolve, 500));
        }
    }
}

app.get("/config", (req, res) => {
    res.json({
        MAX_LEN_CLIENT_SEND: process.env.MAX_LEN_CLIENT_SEND || 35
    });
});

server.listen(process.env.MESSAGE_QUEUE_SERVICE_PORT,"0.0.0.0", () =>
    console.log("Server listening on port ", server.address().address + ":" + process.env.MESSAGE_QUEUE_SERVICE_PORT));
processQueue(); // Chạy process hàng đợi
