'use strict'
const { merge } = require('webpack-merge')
const prodEnv = require('./prod.env')
const fs = require('fs')
const path = require('path')

// Load .env file if it exists
const envFile = path.resolve(__dirname, '../.env')
if (fs.existsSync(envFile)) {
  console.log('Loading .env file from:', envFile)
  const envContent = fs.readFileSync(envFile, 'utf8')
  console.log('Raw .env content:', envContent)
  const envLines = envContent.split('\n')
  envLines.forEach(line => {
    const [key, value] = line.split('=')
    if (key && value) {
      console.log(`Setting ${key.trim()} = ${value.trim().substring(0, 20)}...`)
      process.env[key.trim()] = value.trim()
    }
  })
  console.log('VUE_APP_CESIUM_TOKEN after loading:', process.env.VUE_APP_CESIUM_TOKEN ? 'LOADED' : 'NOT_LOADED')
} else {
  console.log('.env file not found at:', envFile)
}

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  VUE_APP_CESIUM_TOKEN: JSON.stringify(process.env.VUE_APP_CESIUM_TOKEN || '')
})
