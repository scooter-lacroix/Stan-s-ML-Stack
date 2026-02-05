// Package monitoring provides comprehensive event loop monitoring and debugging tools
package monitoring

import (
	"context"
	"fmt"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/diagnostics"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// EventLoopMonitor provides comprehensive event loop monitoring and debugging
type EventLoopMonitor struct {
	ctx               context.Context
	cancel            context.CancelFunc
	config            EventLoopMonitorConfig
	state             EventLoopMonitorState
	metrics           EventLoopMetrics
	diagnostics      *diagnostics.EventLoopDiagnosticContext
	instrumentation  *EventLoopInstrumentation
	profiling         *EventLoopProfiling
	alerts            []EventLoopAlert
	alertChannels     []chan EventLoopAlert
	filters           []EventFilter
	analyzers         []EventAnalyzer
	triggers          []EventTrigger
	handlers          []EventHandler
	mu                sync.RWMutex
	initialized       bool
	startTime         time.Time
	lastCheck         time.Time
	healthScore      float64
	performanceScore float64
	uptime           time.Duration
}

// EventLoopMonitorConfig defines monitor configuration
type EventLoopMonitorConfig struct {
	Enabled               bool
	Interval             time.Duration
	MaxEvents            int
	MaxAlerts           int
	MemoryProfiling     bool
	CPUProfiling        bool
	BlockProfiling     bool
	GCProfiling         bool
	TracingEnabled       bool
	DiagnosticMode      bool
	AlertThresholds     AlertThresholds
	LogLevel            string
	PersistenceEnabled bool
	PersistenceFile     string
	ExportEnabled       bool
	ExportDir           string
	HealthCheckInterval time.Duration
	PerformanceCheckInterval time.Duration
	AlertNotification  bool
}

// EventLoopMonitorState represents monitor state
type EventLoopMonitorState struct {
	Running          bool
	Healthy         bool
	PerformanceGood bool
	DiagnosticMode   bool
	ProfilingActive  bool
	AlertsActive    bool
	ExportActive    bool
	HealthyDuration  time.Duration
	UnhealthyCount   int
	LastHealthCheck  time.Time
	LastPerformanceCheck time.Time
	LastDiagnostic  time.Time
}

// EventLoopMetrics captures comprehensive event loop metrics
type EventLoopMetrics struct {
	EventsProcessed     int64
	EventRate          float64
	FrameRate         float64
	FrameTimeAverage  time.Duration
	FrameTimeMin      time.Duration
	FrameTimeMax      time.Duration
	FrameTimeStdDev   time.Duration
	Goroutines        int
	MemoryUsage       uint64
	GarbageCollected  uint64
	ErrorCount        int64
	WarningCount      int64
	AlertCount        int64
	QueueSize         int
	QueueLength       int64
	ProcessingTime    time.Duration
	IdleTime          time.Duration
	ActivityTime      time.Duration
	PerformanceImpact  string
	HealthStatus      string
	Uptime            time.Duration
}

// AlertThresholds defines alert thresholds
type AlertThresholds struct {
	ErrorRate          float64
	FrameRate         float64
	FrameTime         time.Duration
	MemoryUsage       uint64
	Goroutines        int
	QueueSize         int
	ProcessingTime    time.Duration
	IdleTime          time.Duration
	HealthScore       float64
	PerformanceScore  float64
}

// EventLoopAlert represents an alert condition
type EventLoopAlert struct {
	ID            string
	Type          AlertType
	Level         AlertLevel
	Message       string
	Details       string
	Value         float64
	Threshold     float64
	Timestamp     time.Time
	Duration      time.Duration
	Location      string
	Category      string
	Tags          []string
	Handled       bool
	Resolved      bool
	Resolution    string
	AutoResolved  bool
	Severity      int
}

// AlertType represents alert types
type AlertType string

const (
	AlertTypeError AlertType = "error"
	AlertTypeWarning AlertType = "warning"
	AlertTypeInfo   AlertType = "info"
	AlertTypeHealth AlertType = "health"
	AlertTypePerformance AlertType = "performance"
)

// AlertLevel represents alert levels
type AlertLevel string

const (
	AlertLevelCritical AlertLevel = "critical"
	AlertLevelHigh     AlertLevel = "high"
	AlertLevelMedium   AlertLevel = "medium"
	AlertLevelLow      AlertLevel = "low"
)

// EventLoopInstrumentation provides event loop instrumentation
type EventLoopInstrumentation struct {
	Active        bool
	TraceEnabled  bool
	ProfileEnabled bool
	LogEnabled     bool
	Interval      time.Duration
	EventBuffer   []InstrumentationEvent
	TraceBuffer   []TraceEvent
	MetricsBuffer []MetricEvent
	LastEvent     time.Time
	FirstEvent    time.Time
	EventsCount   int64
	StartTime     time.Time
}

// InstrumentationEvent represents an instrumentation event
type InstrumentationEvent struct {
	Timestamp   time.Time
	Type        string
	Message     string
	Duration    time.Duration
	ModelState  string
	Error       error
	Stack       string
	Location    string
	Tags        []string
	Metadata    map[string]interface{}
}

// TraceEvent represents a trace event
type TraceEvent struct {
	Timestamp   time.Time
	Type        string
	Message     tea.Msg
	Duration    time.Duration
	From        string
	To          string
	Error       error
	Stack       string
}

// MetricEvent represents a metric event
type MetricEvent struct {
	Timestamp time.Time
	Name      string
	Value     float64
	Unit      string
	Tags      []string
}

// EventLoopProfiling provides event loop profiling capabilities
type EventLoopProfiling struct {
	Active        bool
	ProfileDir    string
	CPUProfile    *os.File
	MemProfile    *os.File
	BlockProfile  *os.File
	GCProfile     *os.File
	TraceProfile  *os.File
	ProfileCount  int
	ProfileInterval time.Duration
	StartTime     time.Time
	LastProfile   time.Time
}

// EventFilter represents an event filter
type EventFilter interface {
	Name() string
	Filter(event InstrumentationEvent) bool
}

// EventAnalyzer represents an event analyzer
type EventAnalyzer interface {
	Name() string
	Analyze(events []InstrumentationEvent) AnalysisResult
}

// EventTrigger represents an event trigger
type EventTrigger interface {
	Name() string
	Check(event InstrumentationEvent) bool
}

// EventHandler represents an event handler
type EventHandler interface {
	Name() string
	Handle(event InstrumentationEvent) error
}

// AnalysisResult represents analysis results
type AnalysisResult struct {
	Name        string
	Findings    []AnalysisFinding
	Recommendations []AnalysisRecommendation
	Score       float64
	Confidence float64
	Timestamp   time.Time
}

// AnalysisFinding represents analysis findings
type AnalysisFinding struct {
	Type        string
	Severity    string
	Description string
	Location    string
	Confidence  float64
	Suggestion  string
}

// AnalysisRecommendation represents analysis recommendations
type AnalysisRecommendation struct {
	Priority    int
	Action      string
	Description string
	Benefit     string
	Risk        string
}

// NewEventLoopMonitor creates a new event loop monitor
func NewEventLoopMonitor(config EventLoopMonitorConfig) *EventLoopMonitor {
	ctx, cancel := context.WithCancel(context.Background())

	monitor := &EventLoopMonitor{
		ctx:              ctx,
		cancel:           cancel,
		config:           config,
		state:            EventLoopMonitorState{},
		metrics:          EventLoopMetrics{},
		diagnostics:      diagnostics.NewEventLoopDiagnosticContext(nil),
		instrumentation:  &EventLoopInstrumentation{},
		profiling:        &EventLoopProfiling{},
		alertChannels:     []chan EventLoopAlert{},
		filters:          []EventFilter{},
		analyzers:        []EventAnalyzer{},
		triggers:         []EventTrigger{},
		handlers:         []EventHandler{},
		mu:               sync.RWMutex{},
		initialized:      false,
		startTime:        time.Now(),
		lastCheck:        time.Now(),
		healthScore:      1.0,
		performanceScore: 1.0,
	}

	// Initialize default filters
	monitor.initializeFilters()

	// Initialize default analyzers
	monitor.initializeAnalyzers()

	// Initialize default triggers
	monitor.initializeTriggers()

	// Initialize default handlers
	monitor.initializeHandlers()

	return monitor
}

// initializeFilters initializes default event filters
func (m *EventLoopMonitor) initializeFilters() {
	// Add error filter
	m.filters = append(m.filters, &ErrorFilter{})

	// Add performance filter
	m.filters = append(m.filters, &PerformanceFilter{})

	// Add debug filter
	m.filters = append(m.filters, &DebugFilter{})
}

// initializeAnalyzers initializes default event analyzers
func (m *EventLoopMonitor) initializeAnalyzers() {
	// Add performance analyzer
	m.analyzers = append(m.analyzers, &PerformanceAnalyzer{})

	// Add error analyzer
	m.analyzers = append(m.analyzers, &ErrorAnalyzer{})

	// Add pattern analyzer
	m.analyzers = append(m.analyzers, &PatternAnalyzer{})
}

// initializeTriggers initializes default event triggers
func (m *EventLoopMonitor) initializeTriggers() {
	// Add error trigger
	m.triggers = append(m.triggers, &ErrorTrigger{})

	// Add performance trigger
	m.triggers = append(m.triggers, &PerformanceTrigger{})

	// Add health trigger
	m.triggers = append(m.triggers, &HealthTrigger{})
}

// initializeHandlers initializes default event handlers
func (m *EventLoopMonitor) initializeHandlers() {
	// Add log handler
	m.handlers = append(m.handlers, &LogHandler{})

	// Add alert handler
	m.handlers = append(m.handlers, &AlertHandler{})

	// Add profiling handler
	m.handlers = append(m.handlers, &ProfilingHandler{})
}

// Initialize initializes the monitor
func (m *EventLoopMonitor) Initialize() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initialized {
		return fmt.Errorf("monitor already initialized")
	}

	// Set initial state
	m.state = EventLoopMonitorState{
		Running:          true,
		Healthy:         true,
		PerformanceGood: true,
		DiagnosticMode:   m.config.DiagnosticMode,
		ProfilingActive:  m.config.MemoryProfiling || m.config.CPUProfiling,
		AlertsActive:    m.config.AlertNotification,
		ExportActive:    m.config.ExportEnabled,
		HealthyDuration:  time.Duration(0),
		UnhealthyCount:   0,
		LastHealthCheck:  time.Now(),
		LastPerformanceCheck: time.Now(),
		LastDiagnostic:  time.Now(),
	}

	// Initialize instrumentation
	m.instrumentation.Active = true
	m.instrumentation.TraceEnabled = m.config.TracingEnabled
	m.instrumentation.ProfileEnabled = m.config.MemoryProfiling || m.config.CPUProfiling
	m.instrumentation.LogEnabled = true
	m.instrumentation.Interval = m.config.Interval
	m.instrumentation.StartTime = time.Now()

	// Initialize profiling
	if m.config.MemoryProfiling || m.config.CPUProfiling || m.config.BlockProfiling {
		m.initializeProfiling()
	}

	m.initialized = true
	m.startTime = time.Now()

	// Start monitoring routines
	go m.monitorLoop()
	go m.healthCheckLoop()
	go m.performanceCheckLoop()
	go m.diagnosticCheckLoop()

	return nil
}

// Start starts event loop monitoring
func (m *EventLoopMonitor) Start() error {
	if !m.initialized {
		return fmt.Errorf("monitor not initialized")
	}

	m.state.Running = true
	m.startTime = time.Now()

	return nil
}

// Stop stops event loop monitoring
func (m *EventLoopMonitor) Stop() error {
	m.state.Running = false
	m.cancel()

	// Stop profiling
	if m.profiling.Active {
		m.stopProfiling()
	}

	return nil
}

// RecordEvent records an instrumentation event
func (m *EventLoopMonitor) RecordEvent(event InstrumentationEvent) {
	if !m.state.Running {
		return
	}

	// Apply filters
	for _, filter := range m.filters {
		if !filter.Filter(event) {
			return
		}
	}

	// Add to instrumentation buffer
	m.instrumentation.EventBuffer = append(m.instrumentation.EventBuffer, event)

	// Limit buffer size
	maxEvents := m.config.MaxEvents
	if len(m.instrumentation.EventBuffer) > maxEvents {
		m.instrumentation.EventBuffer = m.instrumentation.EventBuffer[1:]
	}

	// Update metrics
	m.updateMetrics(event)

	// Check triggers
	m.checkTriggers(event)

	// Analyze events
	m.analyzeEvents()

	// Handle event
	m.handleEvent(event)

	// Update last event time
	m.instrumentation.LastEvent = time.Now()
}

// RecordTrace records a trace event
func (m *EventLoopMonitor) RecordTrace(trace TraceEvent) {
	if !m.state.Running {
		return
	}

	m.instrumentation.TraceBuffer = append(m.instrumentation.TraceBuffer, trace)

	// Limit trace buffer size
	if len(m.instrumentation.TraceBuffer) > 1000 {
		m.instrumentation.TraceBuffer = m.instrumentation.TraceBuffer[500:]
	}
}

// RecordMetric records a metric event
func (m *EventLoopMonitor) RecordMetric(metric MetricEvent) {
	if !m.state.Running {
		return
	}

	m.instrumentation.MetricsBuffer = append(m.instrumentation.MetricsBuffer, metric)

	// Limit metrics buffer size
	if len(m.instrumentation.MetricsBuffer) > 1000 {
		m.instrumentation.MetricsBuffer = m.instrumentation.MetricsBuffer[500:]
	}
}

// AddAlertChannel adds an alert channel
func (m *EventLoopMonitor) AddAlertChannel(channel chan EventLoopAlert) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.alertChannels = append(m.alertChannels, channel)
}

// AddFilter adds an event filter
func (m *EventLoopMonitor) AddFilter(filter EventFilter) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.filters = append(m.filters, filter)
}

// AddAnalyzer adds an event analyzer
func (m *EventLoopMonitor) AddAnalyzer(analyzer EventAnalyzer) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.analyzers = append(m.analyzers, analyzer)
}

// AddTrigger adds an event trigger
func (m *EventLoopMonitor) AddTrigger(trigger EventTrigger) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.triggers = append(m.triggers, trigger)
}

// AddHandler adds an event handler
func (m *EventLoopMonitor) AddHandler(handler EventHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.handlers = append(m.handlers, handler)
}

// GetMetrics returns current metrics
func (m *EventLoopMonitor) GetMetrics() EventLoopMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.metrics
}

// GetState returns current monitor state
func (m *EventLoopMonitor) GetState() EventLoopMonitorState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.state
}

// GetAlerts returns current alerts
func (m *EventLoopMonitor) GetAlerts() []EventLoopAlert {
	m.mu.RLock()
	defer m.mu.RUnlock()

	alerts := make([]EventLoopAlert, len(m.alerts))
	copy(alerts, m.alerts)
	return alerts
}

// GetInstrumentationData returns instrumentation data
func (m *EventLoopMonitor) GetInstrumentationData() []InstrumentationEvent {
	m.mu.RLock()
	defer m.mu.RUnlock()

	data := make([]InstrumentationEvent, len(m.instrumentation.EventBuffer))
	copy(data, m.instrumentation.EventBuffer)
	return data
}

// GetTraceData returns trace data
func (m *EventLoopMonitor) GetTraceData() []TraceEvent {
	m.mu.RLock()
	defer m.mu.RUnlock()

	data := make([]TraceEvent, len(m.instrumentation.TraceBuffer))
	copy(data, m.instrumentation.TraceBuffer)
	return data
}

// GetMetricData returns metric data
func (m *EventLoopMonitor) GetMetricData() []MetricEvent {
	m.mu.RLock()
	defer m.mu.RUnlock()

	data := make([]MetricEvent, len(m.instrumentation.MetricsBuffer))
	copy(data, m.instrumentation.MetricsBuffer)
	return data
}

// monitorLoop runs the main monitoring loop
func (m *EventLoopMonitor) monitorLoop() {
	ticker := time.NewTicker(m.config.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.monitorTick()
		}
	}
}

// healthCheckLoop runs health checks
func (m *EventLoopMonitor) healthCheckLoop() {
	ticker := time.NewTicker(m.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.performHealthCheck()
		}
	}
}

// performanceCheckLoop runs performance checks
func (m *EventLoopMonitor) performanceCheckLoop() {
	ticker := time.NewTicker(m.config.PerformanceCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.performPerformanceCheck()
		}
	}
}

// diagnosticCheckLoop runs diagnostic checks
func (m *EventLoopMonitor) diagnosticCheckLoop() {
	if !m.config.DiagnosticMode {
		return
	}

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.performDiagnosticCheck()
		}
	}
}

// monitorTick handles monitoring tick
func (m *EventLoopMonitor) monitorTick() {
	// Update metrics
	m.updateMetricsFromRuntime()

	// Check for alerts
	m.checkForAlerts()

	// Export data if enabled
	if m.config.ExportEnabled {
		m.exportData()
	}

	// Update last check time
	m.lastCheck = time.Now()
}

// performHealthCheck performs health check
func (m *EventLoopMonitor) performHealthCheck() {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check health score
	healthScore := m.calculateHealthScore()

	// Update state
	if healthScore < 0.7 {
		m.state.Healthy = false
		m.state.UnhealthyCount++
		m.state.HealthyDuration = time.Duration(0)
	} else {
		m.state.Healthy = true
		m.state.HealthyDuration += time.Since(m.state.LastHealthCheck)
	}

	m.healthScore = healthScore
	m.state.LastHealthCheck = time.Now()
}

// performPerformanceCheck performs performance check
func (m *EventLoopMonitor) performPerformanceCheck() {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Calculate performance score
	performanceScore := m.calculatePerformanceScore()

	// Update state
	m.state.PerformanceGood = performanceScore > 0.7
	m.performanceScore = performanceScore
	m.state.LastPerformanceCheck = time.Now()
}

// performDiagnosticCheck performs diagnostic check
func (m *EventLoopMonitor) performDiagnosticCheck() {
	// Run diagnostic check
	results := m.diagnostics.RunEventLoopDiagnostics()

	// Store results for analysis
	// This would typically store the results for later retrieval

	m.state.LastDiagnostic = time.Now()
}

// updateMetrics updates metrics from event
func (m *EventLoopMonitor) updateMetrics(event InstrumentationEvent) {
	m.metrics.EventsProcessed++
	m.instrumentation.EventsCount++

	// Update specific metrics based on event type
	switch event.Type {
	case "frame":
		m.updateFrameMetrics(event.Duration)
	case "error":
		m.metrics.ErrorCount++
	case "warning":
		m.metrics.WarningCount++
	case "processing":
		m.metrics.ProcessingTime += event.Duration
	case "idle":
		m.metrics.IdleTime += event.Duration
	case "activity":
		m.metrics.ActivityTime += event.Duration
	}
}

// updateMetricsFromRuntime updates metrics from runtime
func (m *EventLoopMonitor) updateMetricsFromRuntime() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	m.metrics.GarbageCollected = memStats.TotalAlloc - memStats.Alloc
	m.metrics.MemoryUsage = memStats.Alloc
	m.metrics.Goroutines = runtime.NumGoroutine()

	// Calculate event rate
	if m.instrumentation.FirstEvent.IsZero() {
		m.instrumentation.FirstEvent = time.Now()
	}

	duration := time.Since(m.instrumentation.FirstEvent)
	if duration > 0 {
		m.metrics.EventRate = float64(m.instrumentation.EventsCount) / duration.Seconds()
	}
}

// updateFrameMetrics updates frame-specific metrics
func (m *EventLoopMonitor) updateFrameMetrics(frameTime time.Duration) {
	// Calculate frame rate
	if frameTime > 0 {
		frameRate := 1.0 / frameTime.Seconds()
		m.metrics.FrameRate = frameRate
	}

	// Update frame time metrics
	if m.metrics.FrameTimeAverage == 0 {
		m.metrics.FrameTimeAverage = frameTime
	} else {
		// Calculate moving average
		m.metrics.FrameTimeAverage = (m.metrics.FrameTimeAverage + frameTime) / 2
	}

	// Update min/max frame times
	if frameTime < m.metrics.FrameTimeMin || m.metrics.FrameTimeMin == 0 {
		m.metrics.FrameTimeMin = frameTime
	}
	if frameTime > m.metrics.FrameTimeMax || m.metrics.FrameTimeMax == 0 {
		m.metrics.FrameTimeMax = frameTime
	}

	// Calculate standard deviation (simplified)
	// In a real implementation, this would be calculated properly
}

// checkTriggers checks for trigger conditions
func (m *EventLoopMonitor) checkTriggers(event InstrumentationEvent) {
	for _, trigger := range m.triggers {
		if trigger.Check(event) {
			alert := m.createTriggerAlert(trigger, event)
			m.addAlert(alert)
		}
	}
}

// analyzeEvents analyzes recent events
func (m *EventLoopMonitor) analyzeEvents() {
	if len(m.instrumentation.EventBuffer) < 10 {
		return
	}

	// Analyze with each analyzer
	for _, analyzer := range m.analyzers {
		result := analyzer.Analyze(m.instrumentation.EventBuffer)

		// Store result and potentially create alerts
		for _, finding := range result.Findings {
			alert := m.createAnalysisAlert(finding)
			m.addAlert(alert)
		}
	}
}

// handleEvent handles an event
func (m *EventLoopMonitor) handleEvent(event InstrumentationEvent) {
	for _, handler := range m.handlers {
		if err := handler.Handle(event); err != nil {
			// Log handler error
			m.recordError(fmt.Errorf("handler %s failed: %w", handler.Name(), err))
		}
	}
}

// checkForAlerts checks for alert conditions
func (m *EventLoopMonitor) checkForAlerts() {
	// Check error rate
	if m.metrics.ErrorCount > 10 {
		alert := EventLoopAlert{
			ID:        fmt.Sprintf("error_rate_%d", time.Now().Unix()),
			Type:      AlertTypeError,
			Level:     AlertLevelHigh,
			Message:   "High error rate detected",
			Value:     float64(m.metrics.ErrorCount),
			Threshold: 10.0,
			Timestamp: time.Now(),
			Category:  "error",
		}
		m.addAlert(alert)
	}

	// Check frame rate
	if m.metrics.FrameRate < 10.0 {
		alert := EventLoopAlert{
			ID:        fmt.Sprintf("frame_rate_%d", time.Now().Unix()),
			Type:      AlertTypePerformance,
			Level:     AlertLevelMedium,
			Message:   "Low frame rate detected",
			Value:     m.metrics.FrameRate,
			Threshold: 10.0,
			Timestamp: time.Now(),
			Category:  "performance",
		}
		m.addAlert(alert)
	}

	// Check memory usage
	if m.metrics.MemoryUsage > 100*1024*1024 { // 100MB
		alert := EventLoopAlert{
			ID:        fmt.Sprintf("memory_usage_%d", time.Now().Unix()),
			Type:      AlertTypePerformance,
			Level:     AlertLevelHigh,
			Message:   "High memory usage detected",
			Value:     float64(m.metrics.MemoryUsage),
			Threshold: 100 * 1024 * 1024,
			Timestamp: time.Now(),
			Category:  "memory",
		}
		m.addAlert(alert)
	}
}

// createTriggerAlert creates an alert from a trigger
func (m *EventLoopMonitor) createTriggerAlert(trigger EventTrigger, event InstrumentationEvent) EventLoopAlert {
	return EventLoopAlert{
		ID:        fmt.Sprintf("trigger_%s_%d", trigger.Name(), time.Now().Unix()),
		Type:      AlertTypeWarning,
		Level:     AlertLevelMedium,
		Message:   fmt.Sprintf("Trigger %s activated", trigger.Name()),
		Details:   event.Message,
		Value:     1.0,
		Threshold: 1.0,
		Timestamp: time.Now(),
		Category:  "trigger",
		Tags:      []string{trigger.Name()},
	}
}

// createAnalysisAlert creates an alert from analysis finding
func (m *EventLoopMonitor) createAnalysisAlert(finding AnalysisFinding) EventLoopAlert {
	level := AlertLevelLow
	switch finding.Severity {
	case "critical":
		level = AlertLevelCritical
	case "high":
		level = AlertLevelHigh
	case "medium":
		level = AlertLevelMedium
	case "low":
		level = AlertLevelLow
	}

	return EventLoopAlert{
		ID:         fmt.Sprintf("analysis_%s_%d", finding.Type, time.Now().Unix()),
		Type:       AlertType(finding.Type),
		Level:      level,
		Message:    finding.Description,
		Details:    finding.Suggestion,
		Timestamp:  time.Now(),
		Category:   "analysis",
		Tags:       []string{finding.Type},
		Severity:   len(finding.Severity),
	}
}

// addAlert adds an alert
func (m *EventLoopMonitor) addAlert(alert EventLoopAlert) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Limit alerts
	if len(m.alerts) > m.config.MaxAlerts {
		m.alerts = m.alerts[1:]
	}

	// Add alert
	m.alerts = append(m.alerts, alert)
	m.metrics.AlertCount++

	// Send to alert channels
	for _, channel := range m.alertChannels {
		select {
		case channel <- alert:
		default:
			// Channel full, skip
		}
	}
}

// calculateHealthScore calculates health score
func (m *EventLoopMonitor) calculateHealthScore() float64 {
	score := 1.0

	// Check error rate
	if m.metrics.ErrorCount > 0 {
		score -= 0.1 * (float64(m.metrics.ErrorCount) / 100)
	}

	// Check frame rate
	if m.metrics.FrameRate < 30.0 {
		score -= 0.2 * (30.0 - m.metrics.FrameRate) / 30.0
	}

	// Check memory usage
	if m.metrics.MemoryUsage > 100*1024*1024 { // 100MB
		score -= 0.1
	}

	// Check goroutines
	if m.metrics.Goroutines > 50 {
		score -= 0.1 * (float64(m.metrics.Goroutines) / 100)
	}

	// Ensure score is between 0 and 1
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// calculatePerformanceScore calculates performance score
func (m *EventLoopMonitor) calculatePerformanceScore() float64 {
	score := 1.0

	// Check frame rate
	if m.metrics.FrameRate > 0 {
		if m.metrics.FrameRate < 30.0 {
			score *= (m.metrics.FrameRate / 30.0)
		}
	}

	// Check frame time
	if m.metrics.FrameTimeAverage > 100*time.Millisecond {
		score *= (100 * time.Millisecond) / m.metrics.FrameTimeAverage
	}

	// Check memory growth
	if m.metrics.GarbageCollected > 100*1024*1024 { // 100MB
		score *= 0.8
	}

	// Check queue size
	if m.metrics.QueueSize > 50 {
		score *= 0.9
	}

	// Ensure score is between 0 and 1
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// recordError records an error
func (m *EventLoopMonitor) recordError(err error) {
	event := InstrumentationEvent{
		Timestamp: time.Now(),
		Type:      "error",
		Message:   err.Error(),
		Error:     err,
	}
	m.RecordEvent(event)
}

// initializeProfiling initializes profiling
func (m *EventLoopMonitor) initializeProfiling() {
	m.profiling.Active = true
	m.profiling.ProfileDir = m.config.ExportDir
	m.profiling.StartTime = time.Now()
}

// stopProfiling stops profiling
func (m *EventLoopMonitor) stopProfiling() {
	if m.profiling.CPUProfile != nil {
		m.profiling.CPUProfile.Close()
	}
	if m.profiling.MemProfile != nil {
		m.profiling.MemProfile.Close()
	}
	if m.profiling.BlockProfile != nil {
		m.profiling.BlockProfile.Close()
	}
	if m.profiling.GCProfile != nil {
		m.profiling.GCProfile.Close()
	}
	if m.profiling.TraceProfile != nil {
		m.profiling.TraceProfile.Close()
	}
	m.profiling.Active = false
}

// exportData exports monitoring data
func (m *EventLoopMonitor) exportData() {
	// This would export data to files or external services
	// Implementation depends on specific requirements
}

// GetHealthScore returns health score
func (m *EventLoopMonitor) GetHealthScore() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.healthScore
}

// GetPerformanceScore returns performance score
func (m *EventLoopMonitor) GetPerformanceScore() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.performanceScore
}

// IsHealthy returns health status
func (m *EventLoopMonitor) IsHealthy() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.state.Healthy
}

// IsPerformanceGood returns performance status
func (m *EventLoopMonitor) IsPerformanceGood() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.state.PerformanceGood
}

// IsRunning returns running status
func (m *EventLoopMonitor) IsRunning() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.state.Running
}

// GenerateReport generates a comprehensive monitoring report
func (m *EventLoopMonitor) GenerateReport() string {
	var report strings.Builder

	// Header
	report.WriteString("🔍 EVENT LOOP MONITORING REPORT\n")
	report.WriteString("=" * 60 + "\n\n")

	// Current state
	state := m.GetState()
	report.WriteString("📊 CURRENT STATE\n")
	report.WriteString("-" * 40 + "\n")
	report.WriteString(fmt.Sprintf("   Running: %v\n", state.Running))
	report.WriteString(fmt.Sprintf("   Healthy: %v\n", state.Healthy))
	report.WriteString(fmt.Sprintf("   Performance Good: %v\n", state.PerformanceGood))
	report.WriteString(fmt.Sprintf("   Diagnostic Mode: %v\n", state.DiagnosticMode))
	report.WriteString(fmt.Sprintf("   Profiling Active: %v\n", state.ProfilingActive))
	report.WriteString(fmt.Sprintf("   Alerts Active: %v\n", state.AlertsActive))
	report.WriteString(fmt.Sprintf("   Export Active: %v\n", state.ExportActive))
	report.WriteString(fmt.Sprintf("   Healthy Duration: %v\n", state.HealthyDuration))
	report.WriteString(fmt.Sprintf("   Unhealthy Count: %d\n", state.UnhealthyCount))
	report.WriteString("\n")

	// Health and performance metrics
	report.WriteString("📈 HEALTH AND PERFORMANCE\n")
	report.WriteString("-" * 40 + "\n")
	report.WriteString(fmt.Sprintf("   Health Score: %.2f\n", m.GetHealthScore()))
	report.WriteString(fmt.Sprintf("   Performance Score: %.2f\n", m.GetPerformanceScore()))
	report.WriteString(fmt.Sprintf("   Frame Rate: %.1f FPS\n", m.metrics.FrameRate))
	report.WriteString(fmt.Sprintf("   Frame Time Avg: %v\n", m.metrics.FrameTimeAverage))
	report.WriteString(fmt.Sprintf("   Frame Time Min: %v\n", m.metrics.FrameTimeMin))
	report.WriteString(fmt.Sprintf("   Frame Time Max: %v\n", m.metrics.FrameTimeMax))
	report.WriteString(fmt.Sprintf("   Event Rate: %.1f events/s\n", m.metrics.EventRate))
	report.WriteString(fmt.Sprintf("   Goroutines: %d\n", m.metrics.Goroutines))
	report.WriteString(fmt.Sprintf("   Memory Usage: %s\n", formatBytes(m.metrics.MemoryUsage)))
	report.WriteString(fmt.Sprintf("   Garbage Collected: %s\n", formatBytes(m.metrics.GarbageCollected)))
	report.WriteString("\n")

	// Error and alert metrics
	report.WriteString("⚠️  ERROR AND ALERT METRICS\n")
	report.WriteString("-" * 40 + "\n")
	report.WriteString(fmt.Sprintf("   Errors: %d\n", m.metrics.ErrorCount))
	report.WriteString(fmt.Sprintf("   Warnings: %d\n", m.metrics.WarningCount))
	report.WriteString(fmt.Sprintf("   Alerts: %d\n", m.metrics.AlertCount))
	report.WriteString(fmt.Sprintf("   Processing Time: %v\n", m.metrics.ProcessingTime))
	report.WriteString(fmt.Sprintf("   Idle Time: %v\n", m.metrics.IdleTime))
	report.WriteString(fmt.Sprintf("   Activity Time: %v\n", m.metrics.ActivityTime))
	report.WriteString("\n")

	// Alerts
	alerts := m.GetAlerts()
	if len(alerts) > 0 {
		report.WriteString("🚨 CURRENT ALERTS\n")
		report.WriteString("-" * 40 + "\n")
		for _, alert := range alerts {
			report.WriteString(fmt.Sprintf("   [%s] %s: %s\n",
				alert.Level, alert.Type, alert.Message))
			report.WriteString(fmt.Sprintf("      Details: %s\n", alert.Details))
			report.WriteString(fmt.Sprintf("      Timestamp: %s\n", alert.Timestamp.Format("15:04:05")))
			report.WriteString("\n")
		}
	}

	// Instrumentation data
	events := m.GetInstrumentationData()
	if len(events) > 0 {
		report.WriteString("📊 INSTRUMENTATION DATA\n")
		report.WriteString("-" * 40 + "\n")
		report.WriteString(fmt.Sprintf("   Total Events: %d\n", len(events)))
		report.WriteString(fmt.Sprintf("   Event Types:\n"))

		typeCounts := make(map[string]int)
		for _, event := range events {
			typeCounts[event.Type]++
		}

		for eventType, count := range typeCounts {
			report.WriteString(fmt.Sprintf("     %s: %d\n", eventType, count))
		}
		report.WriteString("\n")
	}

	return report.String()
}

// Helper functions

// formatBytes formats bytes human-readable
func formatBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

// Filter implementations

// ErrorFilter filters error events
type ErrorFilter struct{}

func (f *ErrorFilter) Name() string { return "error_filter" }

func (f *ErrorFilter) Filter(event InstrumentationEvent) bool {
	return event.Type == "error" || event.Error != nil
}

// PerformanceFilter filters performance-related events
type PerformanceFilter struct{}

func (f *PerformanceFilter) Name() string { return "performance_filter" }

func (f *PerformanceFilter) Filter(event InstrumentationEvent) bool {
	return event.Type == "frame" || event.Type == "processing" || event.Type == "idle"
}

// DebugFilter filters debug events
type DebugFilter struct{}

func (f *DebugFilter) Name() string { return "debug_filter" }

func (f *DebugFilter) Filter(event InstrumentationEvent) bool {
	return event.Type == "debug"
}

// Analyzer implementations

// PerformanceAnalyzer analyzes performance events
type PerformanceAnalyzer struct{}

func (a *PerformanceAnalyzer) Name() string { return "performance_analyzer" }

func (a *PerformanceAnalyzer) Analyze(events []InstrumentationEvent) AnalysisResult {
	result := AnalysisResult{
		Name:      "Performance Analysis",
		Findings:  []AnalysisFinding{},
		Score:     1.0,
		Confidence: 0.8,
		Timestamp: time.Now(),
	}

	// Analyze frame times
	var frameTimes []time.Duration
	for _, event := range events {
		if event.Type == "frame" {
			frameTimes = append(frameTimes, event.Duration)
		}
	}

	if len(frameTimes) > 0 {
		// Calculate average frame time
		var totalTime time.Duration
		for _, ft := range frameTimes {
			totalTime += ft
		}
		avgFrameTime := totalTime / time.Duration(len(frameTimes))

		// Check for performance issues
		if avgFrameTime > 100*time.Millisecond {
			result.Findings = append(result.Findings, AnalysisFinding{
				Type:        "performance",
				Severity:    "medium",
				Description: fmt.Sprintf("High average frame time: %v", avgFrameTime),
				Location:    "event_loop",
				Confidence:  0.9,
				Suggestion:  "Consider optimizing rendering operations",
			})
			result.Score -= 0.3
		}
	}

	return result
}

// ErrorAnalyzer analyzes error events
type ErrorAnalyzer struct{}

func (a *ErrorAnalyzer) Name() string { return "error_analyzer" }

func (a *ErrorAnalyzer) Analyze(events []InstrumentationEvent) AnalysisResult {
	result := AnalysisResult{
		Name:        "Error Analysis",
		Findings:    []AnalysisFinding{},
		Score:       1.0,
		Confidence:  0.9,
		Timestamp:   time.Now(),
	}

	// Count errors
	errorCount := 0
	for _, event := range events {
		if event.Type == "error" || event.Error != nil {
			errorCount++
		}
	}

	// Analyze error patterns
	if errorCount > 0 {
		result.Findings = append(result.Findings, AnalysisFinding{
			Type:        "error",
			Severity:    "high",
			Description: fmt.Sprintf("High error count detected: %d", errorCount),
			Location:    "event_loop",
			Confidence:  0.9,
			Suggestion:  "Investigate error patterns and implement better error handling",
		})
		result.Score -= 0.5
	}

	return result
}

// PatternAnalyzer analyzes event patterns
type PatternAnalyzer struct{}

func (a *PatternAnalyzer) Name() string { return "pattern_analyzer" }

func (a *PatternAnalyzer) Analyze(events []InstrumentationEvent) AnalysisResult {
	result := AnalysisResult{
		Name:        "Pattern Analysis",
		Findings:    []AnalysisFinding{},
		Score:       1.0,
		Confidence:  0.7,
		Timestamp:   time.Now(),
	}

	// Analyze event patterns
	patternCounts := make(map[string]int)
	for _, event := range events {
		patternCounts[event.Type]++
	}

	// Check for unusual patterns
	for eventType, count := range patternCounts {
		if count > 1000 { // High frequency events
			result.Findings = append(result.Findings, AnalysisFinding{
				Type:        "pattern",
				Severity:    "medium",
				Description: fmt.Sprintf("High frequency of %s events: %d", eventType, count),
				Location:    "event_loop",
				Confidence:  0.8,
				Suggestion:  "Consider optimizing event generation",
			})
			result.Score -= 0.2
		}
	}

	return result
}

// Trigger implementations

// ErrorTrigger triggers on error events
type ErrorTrigger struct{}

func (t *ErrorTrigger) Name() string { return "error_trigger" }

func (t *ErrorTrigger) Check(event InstrumentationEvent) bool {
	return event.Type == "error" || event.Error != nil
}

// PerformanceTrigger triggers on performance events
type PerformanceTrigger struct{}

func (t *PerformanceTrigger) Name() string { return "performance_trigger" }

func (t *PerformanceTrigger) Check(event InstrumentationEvent) bool {
	return event.Type == "frame" && event.Duration > 100*time.Millisecond
}

// HealthTrigger triggers on health events
type HealthTrigger struct{}

func (t *HealthTrigger) Name() string { return "health_trigger" }

func (t *HealthTrigger) Check(event InstrumentationEvent) bool {
	return event.Type == "health" && event.Value < 0.7
}

// Handler implementations

// LogHandler logs events
type LogHandler struct{}

func (h *LogHandler) Name() string { return "log_handler" }

func (h *LogHandler) Handle(event InstrumentationEvent) error {
	// Log the event
	fmt.Printf("[%s] %s: %s\n", event.Timestamp.Format("15:04:05"), event.Type, event.Message)
	return nil
}

// AlertHandler handles alerts
type AlertHandler struct{}

func (h *AlertHandler) Name() string { return "alert_handler" }

func (h *AlertHandler) Handle(event InstrumentationEvent) error {
	// Handle alerts
	if event.Type == "error" || event.Error != nil {
		fmt.Printf("⚠️  ALERT: %s\n", event.Message)
	}
	return nil
}

// ProfilingHandler handles profiling events
type ProfilingHandler struct{}

func (h *ProfilingHandler) Name() string { return "profiling_handler" }

func (h *ProfilingHandler) Handle(event InstrumentationEvent) error {
	// Handle profiling events
	if event.Type == "profile" {
		fmt.Printf("📊 PROFILE: %s\n", event.Message)
	}
	return nil
}

// CreateEventLoopMonitor creates an event loop monitor with default configuration
func CreateEventLoopMonitor() *EventLoopMonitor {
	config := EventLoopMonitorConfig{
		Enabled:               true,
		Interval:             1 * time.Second,
		MaxEvents:            1000,
		MaxAlerts:           100,
		MemoryProfiling:     true,
		CPUProfiling:        true,
		BlockProfiling:     false,
		GCProfiling:         true,
		TracingEnabled:       true,
		DiagnosticMode:      true,
		AlertNotification:    true,
		HealthCheckInterval: 30 * time.Second,
		PerformanceCheckInterval: 60 * time.Second,
	}

	monitor := NewEventLoopMonitor(config)

	// Add alert channel
	alertChan := make(chan EventLoopAlert, 10)
	monitor.AddAlertChannel(alertChan)

	// Start the monitor
	monitor.Initialize()

	return monitor
}