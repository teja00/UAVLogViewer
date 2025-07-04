<template>
    <div>
        <li  v-if="file==null && !sampleLoaded" >
            <a @click="onLoadSample('sample')" class="section"><i class="fas fa-play"></i>  Open Sample </a>
        </li>
        <li v-if="url">
            <a @click="share" class="section"><i class="fas fa-share-alt"></i> {{ shared ? 'Copied to clipboard!' :
                'Share link'}}</a>
        </li>
        <div @click="browse" @dragover.prevent @drop="onDrop" id="drop_zone"
        v-if="file==null && uploadpercentage===-1  && !sampleLoaded">
            <p>Drop *.tlog or *.bin file here or click to browse</p>
            <input @change="onChange" id="choosefile" style="opacity: 0;" type="file">
        </div>
        <!--<b-form-checkbox @change="uploadFile()" class="uploadCheckbox" v-if="file!=null && !uploadStarted"> Upload
        </b-form-checkbox>-->
        <VProgress v-bind:complete="transferMessage"
                   v-bind:percent="uploadpercentage"
                   v-if="uploadpercentage > -1">
        </VProgress>
        <VProgress v-bind:complete="state.processStatus"
                   v-bind:percent="state.processPercentage"
                   v-if="state.processPercentage > -1"
        ></VProgress>
    </div>
</template>
<script>
import VProgress from './SideBarFileManagerProgressBar.vue'
import Worker from '../tools/parsers/parser.worker.js'
import { store } from './Globals'

import { MAVLink20Processor as MAVLink } from '../libs/mavlink'

const worker = new Worker()

worker.addEventListener('message', function (event) {
})

export default {
    name: 'Dropzone',
    data: function () {
        return {
            // eslint-disable-next-line no-undef
            mavlinkParser: new MAVLink(),
            uploadpercentage: -1,
            sampleLoaded: false,
            shared: false,
            url: null,
            transferMessage: '',
            state: store,
            file: null,
            uploadStarted: false,
            backendUrl: 'http://localhost:8000'
        }
    },
    created () {
        this.$eventHub.$on('loadType', this.loadType)
        this.$eventHub.$on('trimFile', this.trimFile)
    },
    beforeDestroy () {
        this.$eventHub.$off('open-sample')
    },
    methods: {
        trimFile () {
            worker.postMessage({ action: 'trimFile', time: this.state.timeRange })
        },
        onLoadSample (file) {
            let url
            if (file === 'sample') {
                this.state.file = 'sample'
                url = require('../assets/vtol.tlog').default
                this.state.logType = 'tlog'
            } else {
                url = file
                // Set the file name for display purposes
                const urlParts = url.split('/')
                this.state.file = urlParts[urlParts.length - 1]
            }

            // Validate URL before making request
            if (!url || typeof url !== 'string') {
                console.error('Invalid URL provided for sample loading:', url)
                alert('Invalid file URL provided')
                return
            }

            const oReq = new XMLHttpRequest()
            console.log(`loading file from ${url}`)

            // Set the log type based on file extension
            this.state.logType = url.indexOf('.tlog') > 0 ? 'tlog' : 'bin'
            if (url.indexOf('.txt') > 0) {
                this.state.logType = 'dji'
            }

            oReq.open('GET', url, true)
            oReq.responseType = 'arraybuffer'

            // Use arrow function to preserve 'this' context
            oReq.onload = (oEvent) => {
                if (oReq.status >= 200 && oReq.status < 300) {
                    const arrayBuffer = oReq.response

                    this.transferMessage = 'Download Done'
                    this.sampleLoaded = true
                    worker.postMessage({
                        action: 'parse',
                        file: arrayBuffer,
                        isTlog: (url.indexOf('.tlog') > 0),
                        isDji: (url.indexOf('.txt') > 0)
                    })
                } else {
                    console.error('Failed to load file:', oReq.status, oReq.statusText)
                    const errorMsg = `Failed to load file: ${oReq.status} ${oReq.statusText}. ` +
                        'Please check the file URL or try uploading a local file instead.'
                    alert(errorMsg)
                    this.uploadpercentage = -1
                    this.transferMessage = ''
                }
            }
            oReq.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    this.uploadpercentage = 100 * e.loaded / e.total
                }
            }
            , false)
            oReq.onerror = (error) => {
                console.error('Error loading file:', error)
                const errorMsg = 'Unable to fetch remote file. ' +
                    'Please check the file URL or try uploading a local file instead.'
                alert(errorMsg)
                this.uploadpercentage = -1
                this.transferMessage = ''
            }

            oReq.send()
        },
        onChange (ev) {
            const fileinput = document.getElementById('choosefile')
            this.process(fileinput.files[0])
        },
        onDrop (ev) {
            // Prevent default behavior (Prevent file from being opened)
            ev.preventDefault()
            if (ev.dataTransfer.items) {
                // Use DataTransferItemList interface to access the file(s)
                for (let i = 0; i < ev.dataTransfer.items.length; i++) {
                    // If dropped items aren't files, reject them
                    if (ev.dataTransfer.items[i].kind === 'file') {
                        const file = ev.dataTransfer.items[i].getAsFile()
                        this.process(file)
                    }
                }
            } else {
                // Use DataTransfer interface to access the file(s)
                for (let i = 0; i < ev.dataTransfer.files.length; i++) {
                    console.log('... file[' + i + '].name = ' + ev.dataTransfer.files[i].name)
                    console.log(ev.dataTransfer.files[i])
                }
            }
        },
        loadType: function (type) {
            worker.postMessage({
                action: 'loadType',
                type: type
            })
        },
        async uploadLogForV2 (file) {
            const formData = new FormData()
            formData.append('file', file)

            try {
                console.log(`Uploading file ${file.name} to backend at ${this.backendUrl}`)
                this.state.v2Processing = true

                const response = await fetch(`${this.backendUrl}/v2/sessions/upload-log`, {
                    method: 'POST',
                    body: formData
                })

                console.log(`Upload response status: ${response.status}`)

                if (!response.ok) {
                    const errorData = await response.json()
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
                }

                const { session_id: sessionId } = await response.json()
                this.state.v2SessionId = sessionId
                console.log(`Started V2 processing with session ID: ${sessionId}`)
                this.$eventHub.$emit('v2-session-created', sessionId)
            } catch (error) {
                console.error('Error uploading log file for V2 analysis:', error)
                // Don't reset v2Processing immediately - let the user retry
                // Only set to false after a delay to prevent UI flickering
                setTimeout(() => {
                    this.state.v2Processing = false
                }, 1000)
                this.$eventHub.$emit('v2-session-error', error.message)
            }
        },
        process: function (file) {
            // Set the file and state variables
            this.state.file = file.name
            this.state.processStatus = 'Pre-processing...'
            this.state.processPercentage = 100
            this.file = file
            this.state.logType = file.name.endsWith('tlog') ? 'tlog' : 'bin'
            if (file.name.endsWith('.txt')) {
                this.state.logType = 'dji'
            }

            // Process file locally using the worker
            const reader = new FileReader()
            reader.onload = function (e) {
                const data = reader.result
                worker.postMessage({
                    action: 'parse',
                    file: data,
                    isTlog: (file.name.endsWith('tlog')),
                    isDji: (file.name.endsWith('txt'))
                })
            }
            reader.readAsArrayBuffer(file)

            // Also upload the file to the V2 backend for chat processing
            // This should happen after setting the file state
            this.$nextTick(() => {
                this.uploadLogForV2(file)
            })
        },
        uploadFile () {
            // This function is not needed for local file processing
            // The app processes files locally without uploading to a server
            console.warn('Upload functionality is disabled. Files are processed locally.')

            // Old upload code commented out to prevent 401 errors
            /*
            this.uploadStarted = true
            this.transferMessage = 'Upload Done!'
            this.uploadpercentage = 0
            const formData = new FormData()
            formData.append('file', this.file)

            const request = new XMLHttpRequest()
            request.onload = () => {
                if (request.status >= 200 && request.status < 400) {
                    this.uploadpercentage = 100
                    this.url = request.responseText
                } else {
                    alert('error! ' + request.status)
                    this.uploadpercentage = 100
                    this.transferMessage = 'Error Uploading'
                    console.log(request)
                }
            }
            request.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    this.uploadpercentage = 100 * e.loaded / e.total
                }
            }
            , false)
            request.open('POST', '/upload')
            request.send(formData)
            */
        },
        fixData (message) {
            if (message.name === 'GLOBAL_POSITION_INT') {
                message.lat = message.lat / 10000000
                message.lon = message.lon / 10000000
                // eslint-disable-next-line
                message.relative_alt = message.relative_alt / 1000
            }
            return message
        },
        browse () {
            document.getElementById('choosefile').click()
        },
        share () {
            const el = document.createElement('textarea')
            el.value = window.location.host + '/#/v/' + this.url
            document.body.appendChild(el)
            el.select()
            document.execCommand('copy')
            document.body.removeChild(el)
            this.shared = true
        },
        downloadFileFromURL (url) {
            const a = document.createElement('a')
            document.body.appendChild(a)
            a.style = 'display: none'
            a.href = url
            a.download = this.state.file + '-trimmed.' + this.state.logType
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)
        }
    },
    mounted () {
        window.addEventListener('message', (event) => {
            if (event.data.type === 'arrayBuffer') {
                worker.postMessage({
                    action: 'parse',
                    file: event.data.data,
                    isTlog: false,
                    isDji: false
                })
            }
        })
        worker.onmessage = (event) => {
            if (event.data.percentage) {
                this.state.processPercentage = event.data.percentage
            } else if (event.data.availableMessages) {
                this.$eventHub.$emit('messageTypes', event.data.availableMessages)
            } else if (event.data.metadata) {
                this.state.metadata = event.data.metadata
            } else if (event.data.messages) {
                this.state.messages = event.data.messages
                this.$eventHub.$emit('messages')
            } else if (event.data.messagesDoneLoading) {
                this.$eventHub.$emit('messagesDoneLoading')
            } else if (event.data.messageType) {
                this.state.messages[event.data.messageType] = event.data.messageList
                this.$eventHub.$emit('messages')
            } else if (event.data.files) {
                this.state.files = event.data.files
                this.$eventHub.$emit('messages')
            } else if (event.data.url) {
                this.downloadFileFromURL(event.data.url)
            }
        }
        const url = document.location.search.split('?file=')[1]
        if (url && url.trim() !== '') {
            const decodedUrl = decodeURIComponent(url)
            console.log('URL parameter found:', decodedUrl)
            // Only load if it's a valid URL or sample file
            const isValidUrl = decodedUrl === 'sample' ||
                (decodedUrl.startsWith('http') &&
                (decodedUrl.includes('.tlog') || decodedUrl.includes('.bin') || decodedUrl.includes('.txt')))
            if (isValidUrl) {
                console.log('Loading file from URL parameter:', decodedUrl)
                this.onLoadSample(decodedUrl)
            } else {
                console.warn('Invalid or unsafe URL parameter ignored:', decodedUrl)
            }
        } else {
            console.log('No file URL parameter found')
        }
    },
    components: {
        VProgress
    }
}
</script>
<style scoped>

    /* NAVBAR */

    #drop_zone {
        padding-top: 25px;
        padding-left: 10px;
        border: 2px dashed #434b52da;
        width: auto;
        height: 100px;
        margin: 20px;
        border-radius: 5px;
        cursor: default;
        background-color: rgba(0, 0, 0, 0);
    }

    #drop_zone:hover {
        background-color: #171e2450;
    }

    .uploadCheckbox {
        margin-left: 20px;
    }

</style>
