<template>
  <div class="chat-interface">
    <!-- Toggle Button -->
    <button
      class="chat-toggle"
      @click="toggleChat"
      :class="{ 'active': isOpen }"
      :disabled="!hasUploadedFile"
      title="Open UAV Chat"
    >
      💬 UAV Chat
    </button>

    <!-- Chat Panel -->
    <div class="chat-panel" v-if="isOpen">
      <div class="chat-header">
        <h4>UAV Log Assistant</h4>
        <button class="close-btn" @click="toggleChat">×</button>
      </div>

      <!-- Chat Section -->
      <div class="chat-section">
        <div class="file-info">
          <span class="file-name">📄 {{ uploadedFileName }}</span>
        </div>

        <!-- Messages -->
        <div class="messages-container" ref="messagesContainer">
          <div
            v-for="(message, index) in messages"
            :key="index"
            class="message"
            :class="{ 'user': message.type === 'user', 'bot': message.type === 'bot' }"
          >
            <div class="message-content" v-html="message.content"></div>
          </div>
          <div v-if="isTyping" class="message bot">
            <div class="message-content typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>

        <!-- Input Area -->
        <div class="input-area">
          <input
            v-model="currentMessage"
            @keyup.enter="sendMessage"
            @keydown.enter.prevent
            placeholder="Ask about your flight data..."
            :disabled="isTyping"
            class="message-input"
          >
          <button
            @click="sendMessage"
            :disabled="!currentMessage.trim() || isTyping"
            class="send-btn"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import { store } from './Globals.js'

export default {
    name: 'ChatInterface',
    data () {
        return {
            isOpen: false,
            currentMessage: '',
            messages: [],
            isTyping: false,
            sessionId: null,
            backendUrl: 'http://localhost:8000', // Backend URL
            state: store,
            initializedFile: null,
            isInitializing: false
        }
    },
    computed: {
        hasUploadedFile () {
            return this.state.file !== null
        },
        uploadedFileName () {
            return this.state.file
        }
    },
    methods: {
        async toggleChat () {
            if (!this.hasUploadedFile) {
                this.isOpen = false
                return
            }

            this.isOpen = !this.isOpen

            if (this.isOpen && this.uploadedFileName !== this.initializedFile) {
                await this.initializeChat()
            }
        },

        async initializeChat () {
            if (this.isInitializing) return
            this.isInitializing = true

            try {
                this.messages = [] // Clear previous messages
                this.addMessage('bot', 'Analyzing your flight data, please wait...')

                const telemetryData = {
                    messages: this.state.messages || {},
                    metadata: {
                        filename: this.uploadedFileName,
                        ...this.state.metadata
                    }
                }

                const response = await axios.post(`${this.backendUrl}/chat`, {
                    message: 'File uploaded - ready for analysis',
                    telemetryData,
                    sessionId: null // Start a new session
                })

                this.sessionId = response.data.sessionId
                this.initializedFile = this.uploadedFileName
                this.messages = [] // Clear "Analyzing" message
                this.addMessage('bot', 'File analysis complete! I am ready to answer your questions.')
            } catch (error) {
                console.error('Chat initialization error:', error)
                this.addMessage('bot', 'Error initializing chat: ' + (error.response?.data?.error || error.message) + '. Please try again.')
            } finally {
                this.isInitializing = false
            }
        },

        async sendMessage () {
            if (!this.currentMessage.trim() || this.isTyping || this.isInitializing) return

            const userMessage = this.currentMessage.trim()
            this.addMessage('user', userMessage)
            this.currentMessage = ''
            this.isTyping = true

            try {
                const response = await axios.post(`${this.backendUrl}/chat`, {
                    message: userMessage,
                    sessionId: this.sessionId
                })

                if (response.data.response) {
                    this.addMessage('bot', response.data.response)
                }
            } catch (error) {
                console.error('Backend communication error:', error)
                this.addMessage('bot', 'Sorry, I encountered an error. Please try again.')
            } finally {
                this.isTyping = false
                this.scrollToBottom()
            }
        },

        addMessage (type, content) {
            this.messages.push({ type, content })
            this.scrollToBottom()
        },

        scrollToBottom () {
            this.$nextTick(() => {
                const container = this.$refs.messagesContainer
                if (container) {
                    container.scrollTop = container.scrollHeight
                }
            })
        }
    },
    watch: {
        'state.file' (newFile, oldFile) {
            if (newFile !== oldFile) {
                this.isOpen = false
                this.initializedFile = null
                this.messages = []
                this.sessionId = null
            }
        }
    }
}
</script>

<style scoped>
.chat-interface {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
}

.chat-toggle {
  background-color: #007bff;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 16px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.chat-toggle.active {
  background-color: #0056b3;
}

.chat-toggle:disabled {
  background-color: #999;
  cursor: not-allowed;
}

.chat-panel {
  width: 350px;
  height: 500px;
  background: #2c3e50; /* Dark blue-grey background */
  border-radius: 10px;
  box-shadow: 0 5px 15px rgba(0,0,0,0.3);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  color: #ecf0f1; /* Light text color */
}

.chat-header {
  padding: 15px;
  background: #34495e; /* Slightly lighter header */
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-header h4 {
  margin: 0;
  color: #ecf0f1;
}

.close-btn {
  background: none;
  border: none;
  color: #ecf0f1;
  font-size: 20px;
  cursor: pointer;
}

.upload-section {
  padding: 20px;
  text-align: center;
  border-bottom: 1px solid #34495e;
}

.upload-area {
  border: 2px dashed #3498db;
  padding: 40px 20px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.upload-area:hover {
  background-color: #34495e;
}

.upload-icon {
  font-size: 40px;
}

.upload-progress {
  margin-top: 15px;
}
.progress-bar {
    background-color: #34495e;
    border-radius: 5px;
    overflow: hidden;
}
.progress-fill {
    height: 10px;
    background-color: #3498db;
    transition: width 0.3s;
}

.chat-section {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.file-info {
  padding: 10px 15px;
  background: #34495e;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}

.file-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 250px;
}

.messages-container {
  flex-grow: 1;
  overflow-y: auto;
  padding: 15px;
}

.message {
  margin-bottom: 15px;
  display: flex;
  flex-direction: column;
}

.message.user {
  align-items: flex-end;
}

.message.bot {
  align-items: flex-start;
}

.message-content {
  background: #34495e;
  padding: 10px 15px;
  border-radius: 15px;
  max-width: 80%;
}

.message.user .message-content {
  background: #2980b9;
  color: white;
}

.message.bot .message-content.typing {
  display: flex;
  align-items: center;
}

.message.bot .message-content.typing span {
  height: 8px;
  width: 8px;
  background: #bdc3c7;
  border-radius: 50%;
  margin: 0 2px;
  animation: typing-blink 1.4s infinite both;
}

.message.bot .message-content.typing span:nth-child(2) {
  animation-delay: 0.2s;
}

.message.bot .message-content.typing span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing-blink {
  0% { opacity: 0.2; }
  20% { opacity: 1; }
  100% { opacity: 0.2; }
}

.input-area {
  padding: 15px;
  display: flex;
  border-top: 1px solid #34495e;
}

.message-input {
  flex-grow: 1;
  padding: 10px;
  border: 1px solid #34495e;
  border-radius: 20px;
  background: #ecf0f1;
  color: #2c3e50;
}

.message-input:focus {
  outline: none;
  border-color: #3498db;
}

.send-btn {
  margin-left: 10px;
  padding: 10px 20px;
  border: none;
  background: #2980b9;
  color: white;
  border-radius: 20px;
  cursor: pointer;
}

.send-btn:disabled {
  background: #95a5a6;
  cursor: not-allowed;
}
</style>
