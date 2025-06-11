<template>
  <div class="chat-interface">
    <!-- Fixed Bubble (Collapsed State) -->
    <div
      v-if="!isExpanded"
      class="chat-bubble"
      :class="{
        'has-unread': hasUnread,
        'is-processing': isProcessing,
        'pulse-intro': showIntroAnimation,
        'ready': isReadyForChat
      }"
      @click="expandChat"
      @mouseenter="showTooltip = true"
      @mouseleave="showTooltip = false"
      tabindex="0"
      @keydown.enter="expandChat"
      @keydown.space.prevent="expandChat"
    >
      <!-- Status Icon -->
      <div class="bubble-icon">
        <div v-if="isProcessing" class="processing-spinner"></div>
        <div v-else class="chat-icon">💬</div>
      </div>

      <!-- Unread Badge -->
      <div v-if="hasUnread && !isProcessing" class="unread-badge">{{ unreadCount }}</div>

      <!-- Tooltip -->
      <transition name="tooltip-fade">
        <div v-if="showTooltip && !isExpanded" class="bubble-tooltip">
          {{ tooltipText }}
        </div>
      </transition>
    </div>

    <!-- Slide-over Panel (Expanded State) -->
    <transition name="slide-panel" @enter="onPanelEnter" @leave="onPanelLeave">
      <div v-if="isExpanded" class="chat-panel">
        <!-- Panel Header -->
        <div class="panel-header">
          <div class="header-content">
            <div class="header-title">
              <h3>UAV Log Assistant</h3>
              <div class="status-indicator">
                <div v-if="isProcessing" class="status processing">
                  <div class="mini-spinner"></div>
                  <span>Processing log...</span>
                </div>
                <div v-else-if="isReadyForChat" class="status ready">
                  <span class="status-dot ready"></span>
                  <span>Ready</span>
                </div>
                <div v-else class="status idle">
                  <span class="status-dot idle"></span>
                  <span>Waiting for log</span>
                </div>
              </div>
            </div>
            <button class="close-button" @click="collapseChat" aria-label="Close chat">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
            </button>
          </div>

          <!-- File Info Bar -->
          <div v-if="hasUploadedFile" class="file-info-bar">
            <div class="file-details">
              <svg class="file-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2"/>
                <polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2"/>
              </svg>
              <span class="file-name">{{ uploadedFileName }}</span>
            </div>
            <div class="file-status">
              <span v-if="state.v2Processing" class="processing-text">
                <div class="processing-dots">
                  <span></span><span></span><span></span>
                </div>
                Processing...
              </span>
              <span v-else-if="isReadyForChat" class="ready-text">
                ✅ Ready for questions
              </span>
            </div>
          </div>
        </div>

        <!-- Messages Area -->
        <div class="messages-area" ref="messagesArea">
          <div class="messages-container">
            <!-- Welcome Message -->
            <div v-if="messages.length === 0 && isReadyForChat" class="welcome-message">
              <div class="welcome-content">
                <div class="welcome-icon">🚁</div>
                <h4>Welcome to UAV Log Assistant</h4>
                <p>Ask me anything about your flight data. Try commands like:</p>
                <div class="command-chips">
                  <span class="chip" @click="sendQuickCommand('Give me a summary of this flight')">📊 Flight Summary</span>
                  <span class="chip" @click="sendQuickCommand('Check for GPS issues')">📡 GPS Issues</span>
                  <span class="chip" @click="sendQuickCommand('Analyze altitude changes')">📈 Altitude Analysis</span>
                  <span class="chip" @click="sendQuickCommand('Show flight modes')">⚙️ Flight Modes</span>
                </div>
              </div>
            </div>

            <!-- Chat Messages -->
            <div
              v-for="(message, index) in messages"
              :key="index"
              class="message-wrapper"
              :class="{ 'user-message': message.type === 'user', 'bot-message': message.type === 'bot' }"
            >
              <div class="message-bubble">
                <div class="message-content" v-html="formatMessage(message.content)"></div>
                <div class="message-timestamp">{{ formatTime(message.timestamp) }}</div>
              </div>
            </div>

            <!-- Typing Indicator -->
            <div v-if="isTyping" class="message-wrapper bot-message">
              <div class="message-bubble typing-bubble">
                <div class="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
                <div class="typing-text">Assistant is thinking...</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Input Area -->
        <div class="input-area">
          <div class="input-container">
            <input
              ref="messageInput"
              v-model="currentMessage"
              @keydown="handleKeyDown"
              @input="handleInput"
              :placeholder="inputPlaceholder"
              :disabled="!canSendMessage"
              class="message-input"
              :class="{ 'disabled': !canSendMessage }"
              autocomplete="off"
            >
            <button
              @click="sendMessage"
              :disabled="!canSendMessage || !currentMessage.trim()"
              class="send-button"
              :class="{ 'sending': isTyping }"
              aria-label="Send message"
            >
              <div v-if="!isTyping" class="send-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
              </div>
              <div v-else class="send-spinner"></div>
            </button>
          </div>

          <!-- Quick Actions -->
          <div v-if="isReadyForChat && !isTyping" class="quick-actions">
            <button class="quick-action" @click="sendQuickCommand('Give me a summary of this flight')">
              <span class="action-icon">📊</span>
              <span class="action-label">Summary</span>
            </button>
            <button class="quick-action" @click="sendQuickCommand('Check for any errors or warnings')">
              <span class="action-icon">⚠️</span>
              <span class="action-label">Issues</span>
            </button>
            <button class="quick-action" @click="sendQuickCommand('Show GPS data quality')">
              <span class="action-icon">📡</span>
              <span class="action-label">GPS</span>
            </button>
            <button class="quick-action" @click="sendQuickCommand('Analyze power consumption')">
              <span class="action-icon">🔋</span>
              <span class="action-label">Power</span>
            </button>
          </div>
        </div>
      </div>
    </transition>

    <!-- Panel backdrop overlay -->
    <div v-if="isExpanded" class="panel-backdrop" @click="collapseChat"></div>
  </div>
</template>

<script>
import axios from 'axios'
import { store } from './Globals.js'

export default {
    name: 'ChatInterface',
    data () {
        return {
            // Core state
            isExpanded: false,
            currentMessage: '',
            messages: [],
            isTyping: false,

            // UI state
            showTooltip: false,
            showIntroAnimation: true,
            hasUnread: false,
            unreadCount: 0,

            // Backend integration (keeping all existing functionality)
            backendUrl: 'http://localhost:8000',
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
            return this.state.file || 'No file selected'
        },
        canSendMessage () {
            return this.isReadyForChat && !this.isProcessing && !this.isTyping
        },
        inputPlaceholder () {
            if (this.isProcessing) return 'Processing your log file...'
            if (!this.isReadyForChat) return 'Upload a log file to start...'
            return 'Ask about your flight data...'
        },
        tooltipText () {
            if (this.isProcessing) return 'Processing log file...'
            if (!this.hasUploadedFile) return 'Upload a log file first'
            if (this.hasUnread) return `${this.unreadCount} new message${this.unreadCount > 1 ? 's' : ''}`
            return 'Ask anything about this log'
        }
    },
    mounted () {
        // Hide intro animation after 3 seconds
        setTimeout(() => {
            this.showIntroAnimation = false
        }, 3000)

        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts()
    },
    beforeDestroy () {
        this.stopProcessingPolling()
    },
    created () {
        // Keep all existing event handlers
        this.$eventHub.$on('v2-session-created', (sessionId) => {
            console.log(`ChatInterface received v2-session-created event with session ID: ${sessionId}`)
            this.isProcessing = true
            this.isReadyForChat = false
            this.messages = []
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

        this.$watch('state.file', (newFile, oldFile) => {
            if (newFile !== oldFile) {
                this.isExpanded = false
                this.isReadyForChat = false
                this.isProcessing = false
                this.state.v2Processing = false
                this.messages = []
                this.hasUnread = false
                this.unreadCount = 0
                this.stopProcessingPolling()
            }
        })
    },
    methods: {
        // UI Methods
        expandChat () {
            if (!this.hasUploadedFile) return
            this.isExpanded = true
            this.hasUnread = false
            this.unreadCount = 0
            this.$nextTick(() => {
                if (this.$refs.messageInput) {
                    this.$refs.messageInput.focus()
                }
            })
        },

        collapseChat () {
            this.isExpanded = false
        },

        onPanelEnter (el) {
            // Panel entrance animation callback
            el.style.transform = 'translateX(100%)'
            // force reflow
            el.offsetHeight // eslint-disable-line no-unused-expressions
            el.style.transform = 'translateX(0)'
        },

        onPanelLeave (el) {
            // Panel exit animation callback
            el.style.transform = 'translateX(100%)'
        },

        // Message handling
        sendQuickCommand (command) {
            this.currentMessage = command
            this.sendMessage()
        },

        formatMessage (content) {
            // Enhanced message formatting
            if (!content) return ''

            // Convert timestamps to clickable links (future enhancement)
            let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>')
            formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>')

            // Convert newlines to <br>
            formatted = formatted.replace(/\n/g, '<br>')

            return formatted
        },

        formatTime (timestamp) {
            if (!timestamp) return ''
            const date = new Date(timestamp)
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        },

        addMessage (type, content) {
            const message = {
                type,
                content,
                timestamp: new Date()
            }
            this.messages.push(message)

            if (type === 'bot' && !this.isExpanded) {
                this.hasUnread = true
                this.unreadCount++
            }

            this.$nextTick(() => {
                this.scrollToBottom()
            })
        },

        scrollToBottom () {
            if (this.$refs.messagesArea) {
                this.$refs.messagesArea.scrollTop = this.$refs.messagesArea.scrollHeight
            }
        },

        // Backend integration (keeping all existing methods)
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
            }, 2000)

            await this.checkProcessingStatus()
        },

        stopProcessingPolling () {
            if (this.processingPollInterval) {
                clearInterval(this.processingPollInterval)
                this.processingPollInterval = null
            }
        },

        async checkProcessingStatus () {
            if (!this.state.v2SessionId) {
                return
            }

            try {
                const response = await axios.get(`${this.backendUrl}/v2/sessions/${this.state.v2SessionId}`)
                const sessionInfo = response.data

                if (sessionInfo.is_processing === false) {
                    this.isProcessing = false
                    this.state.v2Processing = false

                    if (sessionInfo.processing_error) {
                        this.isReadyForChat = false
                        this.messages = []
                        this.addMessage('bot', `Processing failed: ${sessionInfo.processing_error}`)
                    } else {
                        this.isReadyForChat = true
                        this.messages = []
                        this.addMessage('bot', 'Your flight log has been processed and is ready for analysis! Ask me anything about your flight data.')
                    }
                    this.stopProcessingPolling()
                }
            } catch (error) {
                console.error('Status check error:', error)
                this.stopProcessingPolling()
                this.isProcessing = false
                this.isReadyForChat = false
                this.messages = []
                this.addMessage('bot', 'Failed to check processing status. Please try reloading the log file.')
            }
        },

        // Keyboard shortcuts
        setupKeyboardShortcuts () {
            document.addEventListener('keydown', (e) => {
                // Ctrl+Shift+K or Cmd+Shift+K to toggle
                if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'K') {
                    e.preventDefault()
                    if (this.hasUploadedFile) {
                        this.isExpanded ? this.collapseChat() : this.expandChat()
                    }
                }
                // Escape to close
                if (e.key === 'Escape' && this.isExpanded) {
                    this.collapseChat()
                }
            })
        },

        handleKeyDown (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                this.sendMessage()
            }
            // Arrow up to edit last message (future enhancement)
            if (e.key === 'ArrowUp' && !this.currentMessage) {
                // Could implement last message editing here
            }
        },

        handleInput (e) {
            // Could implement slash command detection here
            const value = e.target.value
            if (value.startsWith('/')) {
                // Show command suggestions (future enhancement)
            }
        }
    }
}
</script>

<style scoped>
/* CSS Variables for consistent theming */
.chat-interface {
  --chat-primary: #0B1E36;
  --chat-secondary: #1E2E46;
  --chat-accent: #64e9ff;
  --chat-text: #ecf0f1;
  --chat-text-muted: #95a5a6;
  --chat-success: #27ae60;
  --chat-warning: #f39c12;
  --chat-error: #e74c3c;
  --chat-border: #34495e;
  --chat-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  --chat-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.4);
  --chat-radius: 8px;
  --chat-radius-lg: 16px;

  position: relative;
  pointer-events: none;
  z-index: 1000;
}

/* Chat Bubble (Collapsed State) */
.chat-bubble {
  position: fixed;
  right: 24px;
  bottom: 24px;
  width: 56px;
  height: 56px;
  background: var(--chat-primary);
  border: 2px solid var(--chat-secondary);
  border-radius: 50%;
  cursor: pointer;
  pointer-events: all;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: var(--chat-shadow);
  display: flex;
  align-items: center;
  justify-content: center;
  user-select: none;
  z-index: 1001;
}

.chat-bubble:hover {
  transform: scale(1.05);
  box-shadow: var(--chat-shadow-lg);
  border-color: var(--chat-accent);
}

.chat-bubble.ready {
  border-color: var(--chat-success);
}

.chat-bubble.is-processing {
  border-color: var(--chat-warning);
}

.chat-bubble.has-unread {
  border-color: var(--chat-accent);
  animation: pulse-glow 2s infinite;
}

.chat-bubble.pulse-intro {
  animation: intro-pulse 3s ease-in-out;
}

/* Bubble Icon */
.bubble-icon {
  font-size: 24px;
  color: var(--chat-text);
  display: flex;
  align-items: center;
  justify-content: center;
}

.ready-checkmark {
  color: var(--chat-success);
  font-weight: bold;
  font-size: 20px;
}

.chat-icon {
  filter: grayscale(0.3);
}

/* Processing Spinner */
.processing-spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--chat-secondary);
  border-top: 2px solid var(--chat-warning);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* Unread Badge */
.unread-badge {
  position: absolute;
  top: -4px;
  right: -4px;
  background: var(--chat-accent);
  color: var(--chat-primary);
  border-radius: 12px;
  min-width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  border: 2px solid var(--chat-primary);
}

/* Tooltip */
.bubble-tooltip {
  position: absolute;
  bottom: 70px;
  right: 0;
  background: var(--chat-primary);
  color: var(--chat-text);
  padding: 8px 12px;
  border-radius: var(--chat-radius);
  font-size: 14px;
  white-space: nowrap;
  box-shadow: var(--chat-shadow);
  border: 1px solid var(--chat-secondary);
  pointer-events: none;
}

.bubble-tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  right: 20px;
  border: 6px solid transparent;
  border-top-color: var(--chat-primary);
}

/* Chat Panel (Expanded State) */
.chat-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: min(35vw, 480px);
  height: 100vh;
  background: var(--chat-primary);
  border-left: 1px solid var(--chat-secondary);
  box-shadow: var(--chat-shadow-lg);
  pointer-events: all;
  display: flex;
  flex-direction: column;
  z-index: 1002;
}

/* Panel Header */
.panel-header {
  background: var(--chat-secondary);
  border-bottom: 1px solid var(--chat-border);
}

.header-content {
  padding: 20px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.header-title h3 {
  margin: 0 0 8px 0;
  color: var(--chat-text);
  font-size: 18px;
  font-weight: 600;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.status {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status.processing {
  color: var(--chat-warning);
}

.status.ready {
  color: var(--chat-success);
}

.status.idle {
  color: var(--chat-text-muted);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-dot.ready {
  background: var(--chat-success);
}

.status-dot.idle {
  background: var(--chat-text-muted);
}

.mini-spinner {
  width: 12px;
  height: 12px;
  border: 1.5px solid var(--chat-secondary);
  border-top: 1.5px solid var(--chat-warning);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.close-button {
  background: none;
  border: none;
  color: var(--chat-text-muted);
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s;
}

.close-button:hover {
  color: var(--chat-text);
  background: rgba(255, 255, 255, 0.1);
}

/* File Info Bar */
.file-info-bar {
  padding: 12px 20px;
  background: rgba(0, 0, 0, 0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.file-details {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--chat-text);
}

.file-icon {
  color: var(--chat-accent);
}

.file-name {
  max-width: 200px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file-status {
  display: flex;
  align-items: center;
}

.processing-text {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--chat-warning);
}

.processing-dots {
  display: flex;
  gap: 2px;
}

.processing-dots span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--chat-warning);
  animation: processing-dots 1.4s infinite;
}

.processing-dots span:nth-child(2) { animation-delay: 0.2s; }
.processing-dots span:nth-child(3) { animation-delay: 0.4s; }

.ready-text {
  color: var(--chat-success);
}

/* Messages Area */
.messages-area {
  flex: 1;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--chat-border) transparent;
}

.messages-area::-webkit-scrollbar {
  width: 6px;
}

.messages-area::-webkit-scrollbar-track {
  background: transparent;
}

.messages-area::-webkit-scrollbar-thumb {
  background: var(--chat-border);
  border-radius: 3px;
}

.messages-container {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Welcome Message */
.welcome-message {
  text-align: center;
  padding: 40px 20px;
}

.welcome-content {
  max-width: 300px;
  margin: 0 auto;
}

.welcome-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.welcome-content h4 {
  color: var(--chat-text);
  margin: 0 0 12px 0;
  font-size: 18px;
}

.welcome-content p {
  color: var(--chat-text-muted);
  margin: 0 0 20px 0;
  font-size: 14px;
  line-height: 1.5;
}

.command-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}

.chip {
  background: var(--chat-secondary);
  color: var(--chat-text);
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid var(--chat-border);
}

.chip:hover {
  background: var(--chat-accent);
  color: var(--chat-primary);
  transform: translateY(-1px);
}

/* Message Bubbles */
.message-wrapper {
  display: flex;
  margin-bottom: 12px;
}

.message-wrapper.user-message {
  justify-content: flex-end;
}

.message-wrapper.bot-message {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 85%;
  padding: 12px 16px;
  border-radius: var(--chat-radius-lg);
  position: relative;
  animation: message-appear 0.3s ease-out;
}

.user-message .message-bubble {
  background: var(--chat-accent);
  color: var(--chat-primary);
  border-bottom-right-radius: 4px;
}

.bot-message .message-bubble {
  background: var(--chat-secondary);
  color: var(--chat-text);
  border-bottom-left-radius: 4px;
}

.message-content {
  line-height: 1.5;
  word-wrap: break-word;
}

.message-content code {
  background: rgba(0, 0, 0, 0.2);
  padding: 2px 4px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
}

.message-timestamp {
  font-size: 11px;
  opacity: 0.6;
  margin-top: 4px;
  text-align: right;
}

.bot-message .message-timestamp {
  text-align: left;
}

/* Typing Indicator */
.typing-bubble {
  background: var(--chat-secondary) !important;
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 40px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--chat-text-muted);
  animation: typing-bounce 1.4s infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

.typing-text {
  color: var(--chat-text-muted);
  font-size: 13px;
  font-style: italic;
}

/* Input Area */
.input-area {
  border-top: 1px solid var(--chat-border);
  padding: 20px;
  background: var(--chat-primary);
}

.input-container {
  display: flex;
  gap: 12px;
  align-items: center;
}

.message-input {
  flex: 1;
  background: var(--chat-secondary);
  border: 1px solid var(--chat-border);
  border-radius: 20px;
  padding: 12px 16px;
  color: var(--chat-text);
  font-size: 14px;
  resize: none;
  outline: none;
  transition: all 0.2s;
}

.message-input:focus {
  border-color: var(--chat-accent);
  box-shadow: 0 0 0 2px rgba(100, 233, 255, 0.2);
}

.message-input.disabled {
  background: var(--chat-border);
  color: var(--chat-text-muted);
  cursor: not-allowed;
}

.message-input::placeholder {
  color: var(--chat-text-muted);
}

.send-button {
  width: 44px;
  height: 44px;
  background: var(--chat-accent);
  border: none;
  border-radius: 50%;
  color: var(--chat-primary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.send-button:hover:not(:disabled) {
  background: #4dd9f0;
  transform: scale(1.05);
}

.send-button:disabled {
  background: var(--chat-border);
  color: var(--chat-text-muted);
  cursor: not-allowed;
  transform: none;
}

.send-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  transform: translateX(1px);
}

.send-icon svg {
  width: 18px;
  height: 18px;
  color: inherit;
}

.send-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--chat-primary);
  border-top: 2px solid transparent;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* Quick Actions */
.quick-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.quick-action {
  background: var(--chat-secondary);
  border: 1px solid var(--chat-border);
  border-radius: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  gap: 6px;
  transition: all 0.2s;
  min-width: 80px;
}

.quick-action:hover {
  background: var(--chat-accent);
  border-color: var(--chat-accent);
  color: var(--chat-primary);
  transform: translateY(-1px);
}

.action-icon {
  font-size: 14px;
}

.action-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--chat-text);
}

.quick-action:hover .action-label {
  color: var(--chat-primary);
}

/* Panel Backdrop */
.panel-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.2);
  pointer-events: all;
  z-index: 1001;
}

/* Animations */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: var(--chat-shadow); }
  50% { box-shadow: 0 4px 20px rgba(100, 233, 255, 0.4); }
}

@keyframes intro-pulse {
  0%, 100% { transform: scale(1); }
  10%, 30%, 50% { transform: scale(1.1); }
  20%, 40% { transform: scale(1.05); }
}

@keyframes processing-dots {
  0%, 20%, 80%, 100% { transform: scale(1); opacity: 0.3; }
  50% { transform: scale(1.2); opacity: 1; }
}

@keyframes typing-bounce {
  0%, 20%, 80%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

@keyframes message-appear {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Transition Classes */
.slide-panel-enter-active, .slide-panel-leave-active {
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.slide-panel-enter, .slide-panel-leave-to {
  transform: translateX(100%);
}

.tooltip-fade-enter-active, .tooltip-fade-leave-active {
  transition: opacity 0.2s;
}

.tooltip-fade-enter, .tooltip-fade-leave-to {
  opacity: 0;
}

/* Focus styles for accessibility */
.chat-bubble:focus,
.close-button:focus,
.send-button:focus,
.message-input:focus,
.quick-action:focus,
.chip:focus {
  outline: 2px solid var(--chat-accent);
  outline-offset: 2px;
}
</style>
