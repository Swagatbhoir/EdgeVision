/**
 * EdgeVision ADAS — Cockpit Dashboard JavaScript
 * Real-time telemetry receiver, Web Audio API acoustic alerts, 
 * UI widget controllers, ADAS simulator, and drawer management.
 */

// ==========================================================================
// 1. GLOBAL STATE
// ==========================================================================
const STATE = {
    audioEnabled: false,
    audioMuted: false,
    audioCtx: null,
    telemetryPollInterval: 1000,
    currentAlertLevel: "NORMAL",
    simMode: false,
    logEntries: [],
    logCount: 0,
};

// ==========================================================================
// 2. DOM ELEMENTS
// ==========================================================================
const DOM = {
    // Header
    connDot: document.getElementById('conn-dot'),
    connText: document.getElementById('conn-text'),
    alertBanner: document.getElementById('master-alert-banner'),
    alertHeadline: document.getElementById('alert-headline'),
    alertSubtext: document.getElementById('alert-subtext'),
    alertIconSvg: document.getElementById('alert-icon-svg'),
    chipFps: document.getElementById('chip-fps'),
    chipLatency: document.getElementById('chip-latency'),
    audioToggleBtn: document.getElementById('audio-toggle-btn'),
    audioIcon: document.getElementById('audio-icon'),
    fullscreenBtn: document.getElementById('fullscreen-btn'),

    // FCW Card
    fcwBadge: document.getElementById('fcw-status-badge'),
    valTtc: document.getElementById('val-ttc'),
    valDist: document.getElementById('val-dist'),
    valLat: document.getElementById('val-lat'),
    barDist: document.getElementById('bar-dist'),
    ttcDialFill: document.getElementById('ttc-dial-fill'),

    // LDW Card
    ldwBadge: document.getElementById('ldw-status-badge'),
    ldwVehicleMarker: document.getElementById('ldw-vehicle-marker'),
    driftArrowLeft: document.getElementById('drift-arrow-left'),
    driftArrowRight: document.getElementById('drift-arrow-right'),
    laneOffsetLabel: document.getElementById('lane-offset-label'),
    roadLeftEdge: document.getElementById('road-left-edge'),
    roadRightEdge: document.getElementById('road-right-edge'),

    // FCDW Card
    fcdwBadge: document.getElementById('fcdw-status-badge'),
    fcdwStateText: document.getElementById('fcdw-state-text'),
    fcdwStoppedTimer: document.getElementById('fcdw-stopped-timer'),

    // Objects Table
    objCountBadge: document.getElementById('obj-count-badge'),
    objectsTbody: document.getElementById('objects-tbody'),

    // Drawers
    simDrawerBackdrop: document.getElementById('sim-drawer-backdrop'),
    simDrawer: document.getElementById('sim-drawer'),
    simDrawerBtn: document.getElementById('sim-drawer-btn'),
    closeSimDrawer: document.getElementById('close-sim-drawer'),
    settingsDrawerBackdrop: document.getElementById('settings-drawer-backdrop'),
    settingsDrawer: document.getElementById('settings-drawer'),
    settingsToggleBtn: document.getElementById('settings-toggle-btn'),
    closeSettingsDrawer: document.getElementById('close-settings-drawer'),

    // Settings Controls
    toggleFcw: document.getElementById('toggle-fcw'),
    toggleLdw: document.getElementById('toggle-ldw'),
    toggleFcdw: document.getElementById('toggle-fcdw'),
    sliderCamHeight: document.getElementById('slider-cam-height'),
    sliderCamPitch: document.getElementById('slider-cam-pitch'),
    sliderTtcThresh: document.getElementById('slider-ttc-thresh'),
    valCamHeight: document.getElementById('val-cam-height'),
    valCamPitch: document.getElementById('val-cam-pitch'),
    valTtcThresh: document.getElementById('val-ttc-thresh'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),

    // Log
    logStream: document.getElementById('log-stream'),
    logCount: document.getElementById('log-count'),
    clearLogBtn: document.getElementById('clear-log-btn'),

    // Audio unlock modal
    audioPromptModal: document.getElementById('audio-prompt-modal'),
    enableAudioBtn: document.getElementById('enable-audio-btn'),
};

// ==========================================================================
// 3. WEB AUDIO API — ACOUSTIC ALERT ENGINE
// ==========================================================================
const AudioEngine = {
    ctx: null,
    muted: false,
    lastAlertTime: 0,
    alertCooldown: 2000,

    init() {
        try {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            STATE.audioCtx = this.ctx;
            if (this.ctx.state === 'suspended') {
                DOM.audioPromptModal.style.display = 'flex';
            } else {
                STATE.audioEnabled = true;
            }
        } catch (e) {
            // Browser does not support Web Audio
        }
    },

    resume() {
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume().then(() => {
                STATE.audioEnabled = true;
                DOM.audioPromptModal.style.display = 'none';
            });
        }
    },

    _playTone(freq, duration, volume = 0.3) {
        if (!this.ctx || this.muted || !STATE.audioEnabled) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        gain.gain.setValueAtTime(volume, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration / 1000 + 0.05);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        osc.start(this.ctx.currentTime);
        osc.stop(this.ctx.currentTime + duration / 1000 + 0.05);
    },

    triggerAlert(pattern) {
        const now = Date.now();
        if (now - this.lastAlertTime < this.alertCooldown) return;
        this.lastAlertTime = now;

        switch (pattern) {
            case 'CRITICAL':
                // Rapid emergency double-beep: 1200 Hz + 1600 Hz
                this._playTone(1200, 220, 0.5);
                setTimeout(() => this._playTone(1600, 240, 0.5), 280);
                setTimeout(() => this._playTone(1200, 200, 0.45), 600);
                setTimeout(() => this._playTone(1600, 220, 0.45), 860);
                break;
            case 'WARNING':
                // Sharp single alert: 1000 Hz
                this._playTone(1000, 300, 0.35);
                break;
            case 'ADVISORY':
                // Double low-frequency lane advisory pulse: 800 Hz
                this._playTone(800, 180, 0.25);
                setTimeout(() => this._playTone(800, 180, 0.25), 250);
                break;
            case 'CHIME':
                // Friendly departure chime: 600 Hz + 900 Hz
                this._playTone(600, 180, 0.22);
                setTimeout(() => this._playTone(900, 280, 0.22), 220);
                break;
        }
    },
};

// ==========================================================================
// 4. ALERT BANNER UPDATER
// ==========================================================================
const ALERT_LEVELS = {
    NORMAL: {
        cls: 'alert-normal',
        icon: `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>`,
        headline: 'ALL SYSTEMS NORMAL',
        subtext: 'FORWARD ROAD CLEAR • LANE CENTERED',
    }
};

function applyAlert(alertData) {
    const banner = DOM.alertBanner;
    // Remove all alert classes
    banner.className = 'master-alert-banner';

    if (!alertData || alertData.system === 'NORMAL' || alertData.system === 'NONE') {
        banner.classList.add('alert-normal');
        DOM.alertHeadline.textContent = 'ALL SYSTEMS NORMAL';
        DOM.alertSubtext.textContent = 'FORWARD ROAD CLEAR • LANE CENTERED';
        DOM.alertIconSvg.innerHTML = `<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>`;
        return;
    }

    const level = alertData.level || 'WARNING';
    const system = alertData.system || 'ADAS';
    const message = alertData.message || 'WARNING';
    const subtext = alertData.subtext || '';

    // Banner class
    const levelClassMap = { DANGER: 'alert-danger', WARNING: 'alert-warning', INFO: 'alert-fcdw', CAUTION: 'alert-caution' };
    const systemClassMap = { LDW: 'alert-ldw', FCDW: 'alert-fcdw' };
    const bannerClass = systemClassMap[system] || levelClassMap[level] || 'alert-warning';
    banner.classList.add(bannerClass);

    DOM.alertHeadline.textContent = message;
    DOM.alertSubtext.textContent = subtext || system;

    // Icon
    if (level === 'DANGER') {
        DOM.alertIconSvg.innerHTML = `<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line>`;
    } else if (system === 'LDW') {
        DOM.alertIconSvg.innerHTML = `<line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline>`;
    } else {
        DOM.alertIconSvg.innerHTML = `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>`;
    }

    // Audio
    const soundMap = { CRITICAL: 'CRITICAL', WARNING: 'WARNING', ADVISORY: 'ADVISORY', CHIME: 'CHIME' };
    const soundPattern = alertData.sound_pattern || '';
    if (soundPattern && soundPattern !== 'NONE') {
        AudioEngine.triggerAlert(soundPattern);
    }

    // Log the event
    if (alertData.system && alertData.system !== STATE.currentAlertLevel) {
        addLogEntry(`[${system}]`, message + (subtext ? ` — ${subtext}` : ''), level);
        STATE.currentAlertLevel = alertData.system;
    }
}

// ==========================================================================
// 5. FCW WIDGET UPDATER
// ==========================================================================
function updateFCW(telemetry) {
    const fcw = telemetry.fcw || {};
    const level = fcw.level || 'SAFE';
    const dist = fcw.distance_m || fcw.primary_dist;
    const ttc = fcw.ttc;
    const lat = fcw.lateral_offset_m || fcw.primary_lat;

    // Badge
    const badge = DOM.fcwBadge;
    badge.className = 'badge';
    if (level === 'DANGER') badge.classList.add('badge-danger');
    else if (level === 'WARNING') badge.classList.add('badge-warning');
    else badge.classList.add('badge-safe');
    badge.textContent = level;

    // Values
    DOM.valTtc.textContent = ttc != null ? ttc.toFixed(1) : '--.-';
    DOM.valDist.textContent = dist != null ? dist.toFixed(1) : '--.-';
    DOM.valLat.textContent = lat != null ? (lat >= 0 ? '+' : '') + lat.toFixed(2) : '0.0';

    // TTC Radial Dial (max = 15s)
    const maxTtc = 15.0;
    const ttcFrac = ttc != null ? Math.max(0, Math.min(1, (maxTtc - ttc) / maxTtc)) : 0;
    const circum = 264;
    DOM.ttcDialFill.style.strokeDashoffset = circum * (1 - ttcFrac);
    const dialColors = { DANGER: '#EF4444', WARNING: '#F97316', CAUTION: '#F59E0B', SAFE: '#10B981' };
    DOM.ttcDialFill.style.stroke = dialColors[level] || '#10B981';
    DOM.valTtc.style.color = dialColors[level] || '#F8FAFC';

    // Distance Bar (max = 50m)
    const maxDist = 50;
    const distPct = dist != null ? Math.min(100, (dist / maxDist) * 100) : 0;
    DOM.barDist.style.width = distPct + '%';
    DOM.barDist.className = `stat-bar-fill bar-${level === 'DANGER' ? 'danger' : (level === 'WARNING' ? 'warning' : 'safe')}`;
}

// ==========================================================================
// 6. LDW WIDGET UPDATER
// ==========================================================================
function updateLDW(telemetry) {
    const ldw = telemetry.ldw || {};
    const status = ldw.departure_direction || ldw.alert_direction || 'NORMAL';
    const offset = ldw.offset || 0;  // -1.0 to +1.0

    // Badge
    const badge = DOM.ldwBadge;
    badge.className = 'badge';
    if (status !== 'NORMAL' && status !== 'TRACKING') {
        badge.classList.add('badge-warning');
        badge.textContent = status;
    } else {
        badge.classList.add('badge-safe');
        badge.textContent = 'TRACKING';
    }

    // Vehicle Marker Position (offset -1.0=left edge, 0=center, +1.0=right edge)
    const leftPct = 20;
    const rightPct = 80;
    const markerPct = 50 + offset * (rightPct - leftPct) * 0.5;
    DOM.ldwVehicleMarker.style.left = Math.max(leftPct, Math.min(rightPct, markerPct)) + '%';

    // Lane edge color
    const edgeColor = (status !== 'NORMAL') ? 'var(--color-ldw)' : 'var(--color-safe)';
    DOM.roadLeftEdge.style.background = edgeColor;
    DOM.roadRightEdge.style.background = edgeColor;

    // Drift arrows
    const isDriftLeft = status === 'LEFT' || status === 'DRIFTING LEFT';
    const isDriftRight = status === 'RIGHT' || status === 'DRIFTING RIGHT';
    DOM.driftArrowLeft.className = 'drift-arrow left' + (isDriftLeft ? ' active' : '');
    DOM.driftArrowRight.className = 'drift-arrow right' + (isDriftRight ? ' active' : '');
    const pct = Math.abs(offset * 100).toFixed(0);
    DOM.laneOffsetLabel.textContent = offset === 0 ? '0% DEVIATION' : `${pct}% ${offset < 0 ? 'LEFT' : 'RIGHT'} OF CENTER`;
}

// ==========================================================================
// 7. FCDW WIDGET UPDATER
// ==========================================================================
function updateFCDW(telemetry) {
    const fcdw = telemetry.fcdw || {};
    const state = fcdw.state || 'IDLE';
    const alertActive = fcdw.alert_active || false;
    const stoppedSec = fcdw.stopped_duration || 0;

    const badge = DOM.fcdwBadge;
    badge.className = 'badge';
    if (alertActive) {
        badge.classList.add('badge-info');
        badge.textContent = 'ALERT';
        DOM.fcdwStateText.textContent = 'FRONT VEHICLE MOVING';
        DOM.fcdwStateText.style.color = 'var(--color-fcdw)';
    } else if (state === 'TRACKING_STOPPED') {
        badge.classList.add('badge-info');
        badge.textContent = 'TRACKING';
        DOM.fcdwStateText.textContent = 'MONITORING STOPPED LEAD';
        DOM.fcdwStateText.style.color = 'var(--color-safe)';
    } else {
        badge.classList.add('badge-neutral');
        badge.textContent = 'STANDBY';
        DOM.fcdwStateText.textContent = 'NO QUEUE VEHICLE';
        DOM.fcdwStateText.style.color = 'var(--color-text-muted)';
    }

    const m = Math.floor(stoppedSec / 60);
    const s = Math.floor(stoppedSec % 60);
    DOM.fcdwStoppedTimer.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}s`;
}

// ==========================================================================
// 8. OBJECTS TABLE UPDATER
// ==========================================================================
function updateObjectsTable(telemetry) {
    const objects = telemetry.objects || [];
    const inPath = objects.filter(o => o.in_ahead || o.in_path);

    DOM.objCountBadge.textContent = `${inPath.length} DETECTED`;

    if (inPath.length === 0) {
        DOM.objectsTbody.innerHTML = `<tr class="empty-row"><td colspan="6">No obstacles in forward path</td></tr>`;
        return;
    }

    const riskColors = { DANGER: '#EF4444', WARNING: '#F97316', CAUTION: '#F59E0B', SAFE: '#10B981' };

    DOM.objectsTbody.innerHTML = inPath.slice(0, 8).map(obj => {
        const risk = obj.risk || 'SAFE';
        const color = riskColors[risk] || '#94A3B8';
        const dist = obj.distance_m != null ? obj.distance_m.toFixed(1) + 'm' : obj.distance != null ? obj.distance.toFixed(1) + 'm' : '--';
        const lat = obj.lateral_offset_m != null ? (obj.lateral_offset_m >= 0 ? '+' : '') + obj.lateral_offset_m.toFixed(1) + 'm' : '--';
        const ttc = obj.ttc != null ? obj.ttc.toFixed(1) + 's' : '--';
        return `<tr>
            <td style="color:${color};font-weight:700">#${obj.track_id}</td>
            <td>${(obj.name || '').toUpperCase()}</td>
            <td style="color:${color}">${dist}</td>
            <td>${lat}</td>
            <td>${ttc}</td>
            <td><span style="color:${color};font-weight:700">${risk}</span></td>
        </tr>`;
    }).join('');
}

// ==========================================================================
// 9. PERFORMANCE CHIPS UPDATER
// ==========================================================================
function updatePerformance(telemetry) {
    const fps = telemetry.fps;
    const inf_ms = telemetry.inference_ms;

    if (fps != null) {
        DOM.chipFps.textContent = fps.toFixed(1);
        if (fps >= 24) { DOM.chipFps.className = 'perf-value val-green'; }
        else if (fps >= 15) { DOM.chipFps.className = 'perf-value val-yellow'; }
        else { DOM.chipFps.className = 'perf-value val-red'; }
    }

    if (inf_ms != null) {
        DOM.chipLatency.textContent = inf_ms.toFixed(1) + ' ms';
    }
}

// ==========================================================================
// 10. MAIN TELEMETRY POLLING LOOP
// ==========================================================================
let _consecutiveErrors = 0;

async function pollTelemetry() {
    try {
        const resp = await fetch('/api/telemetry', { signal: AbortSignal.timeout(2000) });
        if (!resp.ok) throw new Error('Bad response ' + resp.status);
        const data = await resp.json();
        _consecutiveErrors = 0;

        // Update connection status
        DOM.connDot.className = 'status-dot pulse-green';
        DOM.connText.textContent = 'SYSTEM ONLINE';

        // Update all widgets
        updatePerformance(data);
        updateFCW(data);
        updateLDW(data);
        updateFCDW(data);
        updateObjectsTable(data);
        applyAlert(data.active_alert);

    } catch (e) {
        _consecutiveErrors++;
        if (_consecutiveErrors >= 3) {
            DOM.connDot.className = 'status-dot';
            DOM.connDot.style.background = '#EF4444';
            DOM.connText.textContent = 'FEED OFFLINE';
        }
    }
}

// ==========================================================================
// 11. SCENARIO SIMULATOR
// ==========================================================================
const SIMULATED_ALERTS = {
    'fcw-danger': {
        system: 'FCW', level: 'DANGER',
        message: 'FORWARD COLLISION WARNING', subtext: 'BRAKE NOW',
        color_bgr: [68, 68, 239], sound_pattern: 'CRITICAL',
    },
    'fcw-warning': {
        system: 'FCW', level: 'WARNING',
        message: 'FORWARD COLLISION WARNING', subtext: 'SLOW DOWN',
        sound_pattern: 'WARNING',
    },
    'ldw-left': {
        system: 'LDW', level: 'WARNING',
        message: 'LANE DEPARTURE WARNING', subtext: 'DRIFTING LEFT',
        sound_pattern: 'ADVISORY',
    },
    'fcdw-depart': {
        system: 'FCDW', level: 'INFO',
        message: 'FRONT VEHICLE MOVING', subtext: 'PROCEED WITH CAUTION',
        sound_pattern: 'CHIME',
    },
    'normal': null,
};

const SIM_TELEMETRY = {
    'fcw-danger': {
        fcw: { level: 'DANGER', distance_m: 6.5, ttc: 1.2, lateral_offset_m: 0.1 },
        ldw: { departure_direction: 'NORMAL', offset: 0 },
        fcdw: { state: 'IDLE', alert_active: false, stopped_duration: 0 },
        objects: [{ track_id: 1, name: 'car', risk: 'DANGER', distance_m: 6.5, ttc: 1.2, lateral_offset_m: 0.1, in_ahead: true }],
    },
    'fcw-warning': {
        fcw: { level: 'WARNING', distance_m: 14.2, ttc: 2.8, lateral_offset_m: 0.0 },
        ldw: { departure_direction: 'NORMAL', offset: 0 },
        fcdw: { state: 'IDLE', alert_active: false, stopped_duration: 0 },
        objects: [{ track_id: 2, name: 'truck', risk: 'WARNING', distance_m: 14.2, ttc: 2.8, lateral_offset_m: 0.0, in_ahead: true }],
    },
    'ldw-left': {
        fcw: { level: 'SAFE', distance_m: null, ttc: null, lateral_offset_m: null },
        ldw: { departure_direction: 'DRIFTING LEFT', offset: -0.75 },
        fcdw: { state: 'IDLE', alert_active: false, stopped_duration: 0 },
        objects: [],
    },
    'fcdw-depart': {
        fcw: { level: 'SAFE', distance_m: 5.2, ttc: null, lateral_offset_m: 0.0 },
        ldw: { departure_direction: 'NORMAL', offset: 0 },
        fcdw: { state: 'DEPARTING', alert_active: true, stopped_duration: 12 },
        objects: [{ track_id: 3, name: 'car', risk: 'CAUTION', distance_m: 5.2, ttc: null, lateral_offset_m: 0.0, in_ahead: true }],
    },
    'normal': {
        fcw: { level: 'SAFE', distance_m: null, ttc: null, lateral_offset_m: null },
        ldw: { departure_direction: 'NORMAL', offset: 0 },
        fcdw: { state: 'IDLE', alert_active: false, stopped_duration: 0 },
        objects: [],
    },
};

let _simTimeout = null;

function triggerSimulation(simKey) {
    if (_simTimeout) { clearTimeout(_simTimeout); _simTimeout = null; }

    const alertData = SIMULATED_ALERTS[simKey];
    const telData = SIM_TELEMETRY[simKey];

    if (!telData) return;

    // Apply simulated state
    applyAlert(alertData);
    updateFCW(telData);
    updateLDW(telData);
    updateFCDW(telData);
    updateObjectsTable(telData);

    if (alertData) {
        addLogEntry('[SIM]', `Simulation: ${alertData.message} — ${alertData.subtext}`, alertData.level);
    }

    // Auto-reset after 6 seconds for non-normal scenarios
    if (simKey !== 'normal') {
        _simTimeout = setTimeout(() => { triggerSimulation('normal'); }, 6000);
    }
}

// ==========================================================================
// 12. EVENT LOG
// ==========================================================================
function addLogEntry(sys, message, level) {
    const now = new Date();
    const timeStr = [now.getHours(), now.getMinutes(), now.getSeconds()].map(v => String(v).padStart(2, '0')).join(':');
    const levelClass = { DANGER: 'log-danger', WARNING: 'log-warning', INFO: 'log-fcdw', ADVISORY: 'log-ldw' }[level] || 'log-neutral';

    STATE.logCount++;
    DOM.logCount.textContent = STATE.logCount;

    const entry = document.createElement('div');
    entry.className = `log-entry ${levelClass}`;
    entry.innerHTML = `<span class="log-time">${timeStr}</span><span class="log-sys" style="color:var(--color-text-muted)">${sys}</span><span class="log-msg">${message}</span>`;

    DOM.logStream.prepend(entry);

    // Keep max 50 entries
    while (DOM.logStream.children.length > 50) {
        DOM.logStream.lastChild.remove();
    }
}

// ==========================================================================
// 13. DRAWER CONTROLLERS
// ==========================================================================
function openDrawer(drawerEl, backdropEl) {
    drawerEl.classList.add('open');
    backdropEl.classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeDrawer(drawerEl, backdropEl) {
    drawerEl.classList.remove('open');
    backdropEl.classList.remove('open');
    document.body.style.overflow = '';
}

// ==========================================================================
// 14. SETTINGS — SLIDER LIVE VALUE DISPLAY
// ==========================================================================
function initSliders() {
    DOM.sliderCamHeight.addEventListener('input', () => {
        DOM.valCamHeight.textContent = parseFloat(DOM.sliderCamHeight.value).toFixed(2) + ' m';
    });
    DOM.sliderCamPitch.addEventListener('input', () => {
        DOM.valCamPitch.textContent = parseFloat(DOM.sliderCamPitch.value).toFixed(1) + '°';
    });
    DOM.sliderTtcThresh.addEventListener('input', () => {
        DOM.valTtcThresh.textContent = parseFloat(DOM.sliderTtcThresh.value).toFixed(1) + ' s';
    });
}

function saveSettings() {
    const payload = {
        fcw_enabled: DOM.toggleFcw.checked,
        ldw_enabled: DOM.toggleLdw.checked,
        fcdw_enabled: DOM.toggleFcdw.checked,
        cam_height_m: parseFloat(DOM.sliderCamHeight.value),
        cam_pitch_deg: parseFloat(DOM.sliderCamPitch.value),
        ttc_danger_threshold: parseFloat(DOM.sliderTtcThresh.value),
    };

    fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    }).then(() => {
        addLogEntry('[CONFIG]', 'Calibration settings applied successfully', 'INFO');
        closeDrawer(DOM.settingsDrawer, DOM.settingsDrawerBackdrop);
    }).catch(() => {
        addLogEntry('[CONFIG]', 'Failed to apply settings — server offline', 'WARNING');
    });
}

// ==========================================================================
// 15. HUD OVERLAY TOGGLES
// ==========================================================================
const toggleStates = {};

document.querySelectorAll('.chip-toggle').forEach(btn => {
    const key = btn.dataset.toggle;
    toggleStates[key] = true;
    btn.addEventListener('click', () => {
        toggleStates[key] = !toggleStates[key];
        btn.classList.toggle('active', toggleStates[key]);
        // Notify backend to toggle HUD layer
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [`hud_${key}`]: toggleStates[key] }),
        }).catch(() => {});
    });
});

// ==========================================================================
// 16. AUDIO MUTE TOGGLE
// ==========================================================================
function toggleMute() {
    AudioEngine.muted = !AudioEngine.muted;
    const icon = DOM.audioToggleBtn.querySelector('svg');
    icon.style.opacity = AudioEngine.muted ? '0.4' : '1.0';
    DOM.audioToggleBtn.title = AudioEngine.muted ? 'Unmute Acoustic Alerts' : 'Mute Acoustic Alerts';
}

// ==========================================================================
// 17. FULLSCREEN TOGGLE
// ==========================================================================
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

// ==========================================================================
// 18. INIT & EVENT BINDINGS
// ==========================================================================
function init() {
    // Start Audio Engine
    AudioEngine.init();

    // Sliders
    initSliders();

    // Event Listeners
    DOM.audioToggleBtn.addEventListener('click', toggleMute);
    DOM.enableAudioBtn.addEventListener('click', () => { AudioEngine.resume(); });
    document.addEventListener('click', () => { if (!STATE.audioEnabled) AudioEngine.resume(); }, { once: true });

    DOM.fullscreenBtn.addEventListener('click', toggleFullscreen);

    DOM.simDrawerBtn.addEventListener('click', () => openDrawer(DOM.simDrawer, DOM.simDrawerBackdrop));
    DOM.closeSimDrawer.addEventListener('click', () => closeDrawer(DOM.simDrawer, DOM.simDrawerBackdrop));
    DOM.simDrawerBackdrop.addEventListener('click', () => closeDrawer(DOM.simDrawer, DOM.simDrawerBackdrop));

    DOM.settingsToggleBtn.addEventListener('click', () => openDrawer(DOM.settingsDrawer, DOM.settingsDrawerBackdrop));
    DOM.closeSettingsDrawer.addEventListener('click', () => closeDrawer(DOM.settingsDrawer, DOM.settingsDrawerBackdrop));
    DOM.settingsDrawerBackdrop.addEventListener('click', () => closeDrawer(DOM.settingsDrawer, DOM.settingsDrawerBackdrop));

    DOM.saveSettingsBtn.addEventListener('click', saveSettings);

    DOM.clearLogBtn.addEventListener('click', () => {
        DOM.logStream.innerHTML = '';
        STATE.logCount = 0;
        DOM.logCount.textContent = '0';
        addLogEntry('[SYSTEM]', 'Event log cleared', 'INFO');
    });

    // Scenario Simulator buttons
    document.querySelectorAll('[data-sim]').forEach(btn => {
        btn.addEventListener('click', () => {
            triggerSimulation(btn.dataset.sim);
            closeDrawer(DOM.simDrawer, DOM.simDrawerBackdrop);
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            closeDrawer(DOM.simDrawer, DOM.simDrawerBackdrop);
            closeDrawer(DOM.settingsDrawer, DOM.settingsDrawerBackdrop);
        }
        if (e.key === 'f' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); toggleFullscreen(); }
        if (e.key === 'm') { toggleMute(); }
    });

    // Start polling
    pollTelemetry();
    setInterval(pollTelemetry, STATE.telemetryPollInterval);

    // Initial log entry
    addLogEntry('[SYSTEM]', 'EdgeVision ADAS Cockpit Dashboard Initialized', 'INFO');
}

document.addEventListener('DOMContentLoaded', init);
