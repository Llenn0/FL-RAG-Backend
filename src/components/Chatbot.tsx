import { useState, type ChangeEvent } from "react";
import "./chat.css";
import axios from "axios";

type ChatState = "idle" | "pending" | "success" | "failed";
const BACKEND_URL = "http://127.0.0.1:5000/";

export default function Chatbot() {
  // Default start message
  const [messages, setMessages] = useState([
    {
      role: "bot",
      content: "Hi there! How can I help you with your documents today?",
    },
  ]);
  const [input, setInput] = useState(""); // Chat input state
  const [status, setStatus] = useState<ChatState>("idle"); // Status of chatbot request as defined above

  // Submit the question to the backend and receive an answer.
  async function handleMessageSend() {
    if (input.length == 0) return;

    // Add user message to the history and bot message placeholder
    let chatMessages = [
      ...messages,
      { role: "user", content: input },
      { role: "bot", content: "Thinking...please wait..." },
    ];
    setInput("");
    setMessages(chatMessages);

    setStatus("pending");
    console.log(input);

    const formData = new FormData();
    formData.append("text", input);

    // Post question to chat backend
    await axios
      .post(BACKEND_URL + "chat", formData, {})
      .then((res) => {
        console.log("Got response ", res.status);
        // Update message state to display response
        let newMessages = [
          ...messages,
          { role: "user", content: input },
          { role: "bot", content: res.data },
        ];
        setMessages(newMessages);
        setStatus("success");
      })
      .catch((err) => {
        console.log(err);
        // Update message state to display error
        let newMessages = [
          ...messages,
          { role: "user", content: input },
          {
            role: "error",
            content:
              "An error occurred while processing your request. Please try again.",
          },
        ];
        setMessages(newMessages);
        setStatus("failed");
      });
  }

  return (
    <section className="chat-window">
      <h2>Chat</h2>
      <div className="chat">
        {messages.map((msg, i) => (
          <div className={msg.role} key={i}>
            <p>{msg.content}</p>
          </div>
        ))}
      </div>
      <br></br>
      <div className="input-area">
        <input
          type="text"
          placeholder="Ask me anything about the stored documents!"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        ></input>
        <button onClick={handleMessageSend}>Send</button>
      </div>
    </section>
  );
}
