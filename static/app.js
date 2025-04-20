// app.js
const chatBox = document.getElementById("chat-box");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");

function addMessage(content, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);

  const avatar = document.createElement("div");
  avatar.classList.add("avatar");
  avatar.innerText = sender === "user" ? "🧑" : "🤖";

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");
  bubble.innerText = content;

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function speak(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(utterance);
}

function sendMessage(message) {
  addMessage(message, "user");
  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message })
  })
    .then(res => res.json())
    .then(data => {
      addMessage(data.response, "bot");
      speak(data.response);
    });
}

sendBtn.addEventListener("click", () => {
  const message = textInput.value.trim();
  if (message) {
    sendMessage(message);
    textInput.value = "";
  }
});

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

micBtn.addEventListener("click", () => {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.start();

  recognition.onresult = function (event) {
    const voiceText = event.results[0][0].transcript;
    sendMessage(voiceText);
  };
});
// Auto welcome message when page loads
window.addEventListener("DOMContentLoaded", () => {
  const welcomeMsg = "Hello, how can I assist you?";
  addMessage(welcomeMsg, "bot");
  speak(welcomeMsg);
});
