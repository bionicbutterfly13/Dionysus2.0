# 🧠 Flux Frontend-First Initialization

## 🚀 One-Command Startup

The Flux Consciousness Emulator is now configured for **frontend-first** experience. When you start the system, the Flux interface automatically opens in your browser.

### Quick Start

```bash
# Frontend-first startup with auto-browser launch
python main.py

# Alternative streamlined launcher
./launch-flux.sh
```

### What Happens Automatically

1. **Backend Services Start** (9127) - Core API and data processing
2. **Frontend Launches** (9243) - Flux consciousness interface
3. **Browser Opens** - Automatic navigation to http://localhost:9243
4. **Consciousness Systems Initialize** - ThoughtSeeds, Archimedes, Daedalus
5. **Real-time Monitoring** - Health checks and error alerts

### Configuration Features

#### Vite Auto-Open (vite.config.ts)
```typescript
server: {
  port: 9243,
  open: true, // Automatically opens browser
  proxy: {
    '/api': { target: 'http://127.0.0.1:9127' }
  }
}
```

#### Launch Script Auto-Open (launch-flux.sh)
```bash
# Automatically opens Flux interface
if command_exists open; then
    open http://localhost:9243  # macOS
elif command_exists xdg-open; then
    xdg-open http://localhost:9243  # Linux  
fi
```

#### Main.py Frontend Priority
- Frontend starts immediately after backend (not last)
- Browser auto-launch with system health monitoring
- Frontend-first server startup order

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| **🌐 Flux Interface** | http://localhost:9243 | **Primary consciousness interface** |
| ⚡ Backend API | http://localhost:9127 | Core data processing |
| 🏛️ Archimedes | http://localhost:8001 | ASI-GO architecture |
| 🔧 Daedalus | http://localhost:8002 | Universal coordinator |

### Debug Mode

```bash
# Enhanced debugging with browser alerts
python main.py --debug
```

### Development Experience

The frontend-first approach means:
- ✅ **Instant visual feedback** - Interface opens immediately
- ✅ **Auto-reload** - Changes reflected in real-time
- ✅ **Error visibility** - Browser alerts for system issues
- ✅ **Health monitoring** - Continuous connectivity checks
- ✅ **Graceful shutdown** - Ctrl+C stops all services cleanly

### For Development

```bash
# Just frontend (if backend already running)
cd frontend && npm run dev

# Just backend (if frontend already running)  
cd backend && python main.py

# Full system with monitoring
python main.py --debug
```

The system is designed so that **starting the main.py automatically gives you the full Flux experience** with the browser interface ready to use immediately.