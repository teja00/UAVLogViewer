// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App.vue'
import router from './router'

// Importing Bootstrap Vue
import BootstrapVue from 'bootstrap-vue'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

// Using imported components
import VueRouter from 'vue-router'

Vue.use(VueRouter)
Vue.use(BootstrapVue)

Vue.config.productionTip = false

Vue.prototype.$eventHub = new Vue() // Global event bus

// Add comprehensive global error handling to prevent page reloads
Vue.config.errorHandler = function (err, vm, info) {
    console.error('Vue error caught by global handler:', err, info)
    // Log the error but don't let it crash the app
    // Return false to prevent Vue from showing default error message
    return false
}

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function (event) {
    console.error('Unhandled promise rejection:', event.reason)
    // Prevent the default behavior (which might cause page reload)
    event.preventDefault()
    return false
})

// Handle general JavaScript errors
window.addEventListener('error', function (event) {
    console.error('Global JavaScript error:', event.error)
    // Prevent error from bubbling up and potentially causing page reload
    event.preventDefault()
    return false
})

// Disable webpack hot module replacement in production-like environment
if (typeof module !== 'undefined' && module.hot) {
    module.hot.accept()
    // Disable hot reload for better stability
    module.hot.dispose = function () {
        console.log('Module disposed - preventing automatic reload')
        return false
    }
}

/* eslint-disable no-new */
new Vue({
    el: '#app',
    router,
    components: { App },
    template: '<App/>'
})
