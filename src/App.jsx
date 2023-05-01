import { useState } from 'react'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator } from '@chatscope/chat-ui-kit-react';
import axios from 'axios'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    {
      message: "Hello, I'm Chatbot! I can provide solutions to your problems!",
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);


  
  async function handleSend(inputText) {
    setMessages((prevMessages) => [
      ...prevMessages,
      { message: inputText, sender: "user" },
    ]); 

    setIsTyping(true); // start typing indicator

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ fin: inputText }),
      });

      const data = await response.json();

      setTimeout(() => {
        setIsTyping(false); // stop typing indicator after timeout

        if (data && data.response) {
          setMessages((prevMessages) => [
            ...prevMessages,
            { message: data.response, sender: "bot" },
          ]);
        } else {
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              message: "I'm sorry, I don't know the answer to that.",
              sender: "bot",
            },
          ]);
        }
      }, 2000); // wait for 2 seconds
    } catch (error) {
      console.error(error);
      setIsTyping(false);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          message: "Oops! Something went wrong. Please try again later.",
          sender: "bot",
        },
      ]);
    }
  }



  return (
    <div className="App">
      <div style={{ position:"relative", height: "500px", width: "700px"  }}>
         <h2>CHATBOT</h2>
        <MainContainer>
          <ChatContainer>       
            <MessageList 
              scrollBehavior="smooth" 
              typingIndicator={isTyping ? <TypingIndicator content="ChatBot is typing" /> : null}
            >
              {messages.map((message, i) => {
                const lastele = messages[messages.length-1];
                const fin = lastele['message'];
                return <Message key={i} model={message} />
              })}
            </MessageList>
            <MessageInput placeholder="Please enter your problem" onSend={(inputText) => handleSend(inputText)} />       
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  )
}

export default App