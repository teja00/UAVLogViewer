<template>
  <div class="chat-interface">
    <!-- Toggle Button -->
    <button
      class="chat-toggle"
      @click="toggleChat"
      :class="{ 'active': isOpen }"
      :disabled="!hasUploadedFile"
      title="Open UAV Chat"
      v-show="!isOpen"
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
          <span v-if="state.v2Processing" class="processing-indicator">
            <span class="spinner"></span>
            Processing...
          </span>
          <span v-else-if="isReadyForChat" class="ready-indicator">
            ✅ Ready
          </span>
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
            :placeholder="isProcessing ? 'Processing your log file...' : 'Ask about your flight data...'"
            :disabled="isTyping || !isReadyForChat || isProcessing"
            class="message-input"
            :class="{ 'processing': isProcessing }"
          >
          <button
            @click="sendMessage"
            :disabled="!currentMessage.trim() || isTyping || !isReadyForChat || isProcessing"
            class="send-btn"
            :class="{ 'processing': isProcessing }"
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
            backendUrl: 'http://localhost:8000', // Backend URL
            state: store,
            isReadyForChat: false,
            isProcessing: false,
            processingPollInterval: null
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
    created () {
        this.$eventHub.$on('v2-session-created', (sessionId) => {
            console.log(`ChatInterface received v2-session-created event with session ID: ${sessionId}`)
            this.isProcessing = true
            this.isReadyForChat = false
            this.messages = [] // Clear any previous messages
            this.addMessage('bot', 'Your flight log is being processed. Please wait while we analyze your data...')
            this.startProcessingPolling()
        })

        this.$eventHub.$on('v2-session-error', (errorMessage) => {
            this.isProcessing = false
            this.isReadyForChat = false
            this.state.v2Processing = false
            this.messages = []
            this.addMessage('bot', `An error occurred: ${errorMessage}`)
            this.stopProcessingPolling()
        })

        // Handle file changes to reset the chat state
        this.$watch('state.file', (newFile, oldFile) => {
            if (newFile !== oldFile) {
                this.isOpen = false
                this.isReadyForChat = false
                this.isProcessing = false
                this.state.v2Processing = false
                this.messages = []
                this.stopProcessingPolling()
            }
        })
    },
    beforeDestroy () {
        this.stopProcessingPolling()
    },
    methods: {
        toggleChat () {
            if (!this.hasUploadedFile) {
                this.isOpen = false
                return
            }
            this.isOpen = !this.isOpen
            if (this.isOpen && !this.isReadyForChat && this.hasUploadedFile) {
                this.messages = [{ type: 'bot', content: 'Data is loading in the backend...' }]
            }
        },

        async sendMessage () {
            if (!this.currentMessage.trim() || this.isTyping || !this.isReadyForChat || this.isProcessing) return

            const userMessage = this.currentMessage.trim()
            this.addMessage('user', userMessage)
            this.currentMessage = ''
            this.isTyping = true

            try {
                if (!this.state.v2SessionId) {
                    throw new Error('V2 Session ID is not available.')
                }

                const response = await axios.post(`${this.backendUrl}/v2/chat`, {
                    message: userMessage,
                    // eslint-disable-next-line camelcase
                    session_id: this.state.v2SessionId
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

        async startProcessingPolling () {
            if (this.processingPollInterval) {
                clearInterval(this.processingPollInterval)
            }

            this.processingPollInterval = setInterval(async () => {
                await this.checkProcessingStatus()
            }, 2000) // Poll every 2 seconds

            // Also check immediately
            await this.checkProcessingStatus()
        },

        stopProcessingPolling () {
            if (this.processingPollInterval) {
                clearInterval(this.processingPollInterval)
                this.processingPollInterval = null
            }
        },

        async checkProcessingStatus () {
            if (!this.state.v2SessionId) return

            try {
                const response = await axios.get(`${this.backendUrl}/v2/sessions/${this.state.v2SessionId}`)
                const sessionInfo = response.data

                console.log('Session status:', sessionInfo)

                if (!sessionInfo.is_processing && sessionInfo.has_dataframes) {
                    // Processing is complete and data is available
                    this.isProcessing = false
                    this.isReadyForChat = true
                    this.state.v2Processing = false
                    this.stopProcessingPolling()

                    // Update the bot message
                    if (this.messages.length > 0 && this.messages[0].type === 'bot') {
                        this.messages[0].content = 'Your flight log has been processed successfully and you can now ask questions about your flight data.'
                    } else {
                        this.addMessage('bot', 'Your flight log has been processed successfully and you can now ask questions about your flight data.')
                    }
                } else if (!sessionInfo.is_processing && sessionInfo.processing_error) {
                    // Processing failed
                    this.isProcessing = false
                    this.isReadyForChat = false
                    this.state.v2Processing = false
                    this.stopProcessingPolling()
                    this.addMessage('bot', `Processing failed: ${sessionInfo.processing_error}`)
                }
                // If still processing, continue polling
            } catch (error) {
                console.error('Error checking processing status:', error)
                // Don't stop polling on network errors, just log them
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
    max-width: 200px;
}

.processing-indicator {
  display: flex;
  align-items: center;
  color: #f39c12;
  font-size: 12px;
  font-weight: bold;
}

.ready-indicator {
  color: #27ae60;
  font-size: 12px;
  font-weight: bold;
}

.spinner {
  width: 12px;
  height: 12px;
  border: 2px solid #34495e;
  border-top: 2px solid #f39c12;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-right: 5px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
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
  max-width: 80%;
  padding: 10px 15px;
  border-radius: 15px;
  word-wrap: break-word;
}

.message.user .message-content {
  background-color: #3498db;
  color: white;
}

.message.bot .message-content {
  background-color: #34495e;
  color: #ecf0f1;
}

.input-area {
  padding: 15px;
  border-top: 1px solid #34495e;
  display: flex;
  gap: 10px;
}

.message-input {
  flex-grow: 1;
  padding: 10px 15px;
  border: 1px solid #34495e;
  border-radius: 20px;
  background-color: #2c3e50;
  color: #ecf0f1;
  outline: none;
}

.message-input:focus {
  border-color: #3498db;
}

.message-input.processing {
  background-color: #2c3e50;
  border-color: #95a5a6;
  color: #95a5a6;
  cursor: not-allowed;
}

.send-btn {
  padding: 10px 20px;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  font-weight: bold;
}

.send-btn:hover:not(:disabled) {
  background-color: #2980b9;
}

.send-btn:disabled {
  background-color: #7f8c8d;
  cursor: not-allowed;
}

.send-btn.processing {
  background-color: #95a5a6;
  cursor: not-allowed;
}

/* Typing indicator */
.typing {
  display: flex;
  align-items: center;
  padding: 10px 15px;
}

.typing span {
  height: 8px;
  width: 8px;
  border-radius: 50%;
  background-color: #95a5a6;
  display: inline-block;
  margin-right: 5px;
  animation: typing 1.4s infinite;
}

.typing span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-10px);
  }
}

</style>
