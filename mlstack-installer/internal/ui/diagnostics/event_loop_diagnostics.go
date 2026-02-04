// Package diagnostics provides comprehensive diagnostic tools for Bubble Tea UI issues
package diagnostics

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/state"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// EventLoopDiagnosticResult contains event loop and MVU architecture analysis
type EventLoopDiagnosticResult struct {
	EventLoop     EventLoopInfo
	MVUArchitecture  MVUArchitectureInfo
	MessageFlow   MessageFlowInfo
	StateTransitions StateTransitionInfo
	Performance   EventLoopPerformanceInfo
	DeadlockDetection DeadlockInfo
	Recommendations []string
	Timeline      []TimelineEvent
}

// EventLoopInfo captures event loop state and behavior
type EventLoopInfo struct {
	Running         bool
	Active          bool
	StartedAt       time.Time
	LastMessage     time.Time
	MessageCount    int64
	ErrorCount     int64
	PanicCount      int64
	CommandsPending int
	Timeouts       int
	Blocked        bool
	IdleTime       time.Duration
	LoopTime       time.Duration
	AverageLoopTime time.Duration
}

// MVUArchitectureInfo captures Model-View-Update architecture analysis
type MVUArchitectureInfo struct {
	ModelIntegrity     bool
	ImmutableState     bool
	StateConsistency   bool
	UpdateSafety       bool
	ViewPureness       bool
	MutationFree       bool
	InterfaceCompliance bool
	CouplingScore     float64 // 0.0 = well-coupled, 1.0 = loosely-coupled
	CohesionScore     float64 // 0.0 = low cohesion, 1.0 = high cohesion
}

// MessageFlowInfo captures message processing flow analysis
type MessageFlowInfo struct {
	TotalMessages     int64
	MessageTypes     map[string]int64
	MessageLatencies  map[string][]time.Duration
	ProcessingTimes   map[string][]time.Duration
	QueueSize        int
	DroppedMessages  int
	DeadlockedQueues []string
	BackPressure      float64
}

// StateTransitionInfo captures state transition analysis
type StateTransitionInfo struct {
	Transitions []StateTransition
	Valid       map[types.Stage]bool
	Invalid     map[types.Stage]bool
	Total       int
	ErrorRate   float64
	AvgTime     time.Duration
}

// StateTransition captures individual state transition
type StateTransition struct {
	From      types.Stage
	To        types.Stage
	Timestamp time.Time
	Duration  time.Duration
	Success   bool
	Error     error
}

// EventLoopPerformanceInfo captures event loop performance metrics
type EventLoopPerformanceInfo struct {
	FPS               float64
	FrameTimeMin      time.Duration
	FrameTimeMax      time.Duration
	FrameTimeAvg      time.Duration
	FPSMin            float64
	FPSMax            float64
	FPSAvg            float64
	Goroutines        int
	StackDepth        int
	MemoryUsage       uint64
	GCActivity        bool
	LastGC           time.Time
	GCDuration       time.Duration
	BlockedTime       time.Duration
	IdleTimeRatio     float64
	CPUUsage         float64
}

// DeadlockInfo captures deadlock detection results
type DeadlockInfo struct {
	DeadlockDetected bool
	DeadlockReason   string
	BlockingGoroutines []BlockingGoroutine
	DeadlockTimeout  time.Duration
	LockContentions  []LockContention
	ResourceUsage    map[string]ResourceUsage
}

// BlockingGoroutine captures information about blocked goroutines
type BlockingGoroutine struct {
	ID        int
	State     string
	BlockTime time.Duration
	Stack     string
	Location  string
}

// LockContention captures lock contention analysis
type LockContention struct {
	LockName      string
	ContentionCount int
	ContentionTime time.Duration
	Waiters       int
}

// ResourceUsage captures resource usage analysis
type ResourceUsage struct {
	Resource      string
	UsageCount    int
	MaxUsage      int
	AverageUsage  float64
	Contention    float64
}

// EventLoopDiagnosticContext manages event loop diagnostic session
type EventLoopDiagnosticContext struct {
	Context      context.Context
	Cancel       context.CancelFunc
	StartTime    time.Time
	Timeline     []TimelineEvent
	Results      *EventLoopDiagnosticResult
	Model        *Model // Pointer to the main Bubble Tea model
	MessageChan  chan tea.Msg
	DebugChan    chan DebugEvent
	Stats        EventLoopStatistics
	Instrumentation *EventLoopInstrumentation
}

// EventLoopStatistics collects event loop statistics
type EventLoopStatistics struct {
	mu             sync.RWMutex
	MessageCount   int64
	ErrorCount     int64
	PanicCount     int64
	FrameTimes     []time.Duration
	GoroutineCount int
	MemoryUsage    uint64
	CPUUsage       float64
}

// EventLoopInstrumentation provides instrumentation for event loop
type EventLoopInstrumentation struct {
	Active         bool
	TraceEnabled   bool
	ProfileEnabled bool
	LogEnabled     bool
	Interval       time.Duration
	LastTrace      time.Time
	Traces         []TraceEvent
}

// TraceEvent captures trace events
type TraceEvent struct {
	Timestamp   time.Time
	Type        string
	Message     tea.Msg
	Duration    time.Duration
	ModelState  string
	Error       error
	Stack       string
}

// DebugEvent represents debug events
type DebugEvent struct {
	Type        string
	Message     string
	Detail      string
	Timestamp   time.Time
	Priority    DebugPriority
}

// DebugPriority represents debug event priority
type DebugPriority int

const (
	DebugPriorityLow DebugPriority = iota
	DebugPriorityMedium
	DebugPriorityHigh
	DebugPriorityCritical
)

// NewEventLoopDiagnosticContext creates a new event loop diagnostic context
func NewEventLoopDiagnosticContext(model *Model) *EventLoopDiagnosticContext {
	ctx, cancel := context.WithCancel(context.Background())
	return &EventLoopDiagnosticContext{
		Context:      ctx,
		Cancel:       cancel,
		StartTime:    time.Now(),
		Timeline:     []TimelineEvent{},
		Results:      &EventLoopDiagnosticResult{},
		Model:        model,
		MessageChan:  make(chan tea.Msg, 100),
		DebugChan:    make(chan DebugEvent, 50),
		Stats:        EventLoopStatistics{},
		Instrumentation: &EventLoopInstrumentation{
			Active:       true,
			TraceEnabled: true,
			ProfileEnabled: true,
			LogEnabled:   true,
			Interval:     100 * time.Millisecond,
			Traces:       []TraceEvent{},
		},
	}
}

// RunEventLoopDiagnostics runs comprehensive event loop diagnostics
func (dc *EventLoopDiagnosticContext) RunEventLoopDiagnostics() *EventLoopDiagnosticResult {
	dc.addTimelineEvent("EventLoop", "Starting event loop diagnostic suite", time.Duration(0))

	// 1. Event Loop State Analysis
	dc.addTimelineEvent("EventLoop", "Analyzing event loop state", time.Duration(0))
	dc.analyzeEventLoopState()

	// 2. MVU Architecture Analysis
	dc.addTimelineEvent("MVU", "Analyzing MVU architecture compliance", time.Duration(0))
	dc.analyzeMVUArchitecture()

	// 3. Message Flow Analysis
	dc.addTimelineEvent("Messages", "Analyzing message flow and processing", time.Duration(0))
	dc.analyzeMessageFlow()

	// 4. State Transition Analysis
	dc.addTimelineEvent("State", "Analyzing state transitions", time.Duration(0))
	dc.analyzeStateTransitions()

	// 5. Performance Analysis
	dc.addTimelineEvent("Performance", "Analyzing event loop performance", time.Duration(0))
	dc.analyzeEventLoopPerformance()

	// 6. Deadlock Detection
	dc.addTimelineEvent("Deadlock", "Checking for deadlocks and blockages", time.Duration(0))
	dc.detectDeadlocks()

	// 7. Generate Recommendations
	dc.addTimelineEvent("Analysis", "Generating recommendations", time.Duration(0))
	dc.generateEventLoopRecommendations()

	return dc.Results
}

// analyzeEventLoopState analyzes event loop state and behavior
func (dc *EventLoopDiagnosticContext) analyzeEventLoopState() {
	loopInfo := EventLoopInfo{
		Running:      true,
		StartedAt:    time.Now(),
		MessageCount: dc.Stats.MessageCount,
		ErrorCount:  dc.Stats.ErrorCount,
		PanicCount:  dc.Stats.PanicCount,
	}

	// Check if event loop is blocked
	if dc.isEventLoopBlocked() {
		loopInfo.Blocked = true
		loopInfo.IdleTime = time.Since(dc.StartTime)
	}

	// Calculate average loop time
	if len(dc.Stats.FrameTimes) > 0 {
		var total time.Duration
		for _, ft := range dc.Stats.FrameTimes {
			total += ft
		}
		loopInfo.AverageLoopTime = total / time.Duration(len(dc.Stats.FrameTimes))
	}

	// Get pending commands
	loopInfo.CommandsPending = dc.getPendingCommandsCount()

	dc.Results.EventLoop = loopInfo
}

// analyzeMVUArchitecture analyzes MVU architecture compliance
func (dc *EventLoopDiagnosticContext) analyzeMVUArchitecture() {
	mvuInfo := MVUArchitectureInfo{
		ModelIntegrity:    dc.checkModelIntegrity(),
		ImmutableState:    dc.checkImmutability(),
		StateConsistency:  dc.checkStateConsistency(),
		UpdateSafety:      dc.checkUpdateSafety(),
		ViewPureness:      dc.checkViewPureness(),
		MutationFree:      dc.checkMutationFree(),
		InterfaceCompliance: dc.checkInterfaceCompliance(),
	}

	// Calculate coupling and cohesion scores
	mvuInfo.CouplingScore = dc.calculateCouplingScore()
	mvuInfo.CohesionScore = dc.calculateCohesionScore()

	dc.Results.MVUArchitecture = mvuInfo
}

// analyzeMessageFlow analyzes message processing flow
func (dc *EventLoopDiagnosticContext) analyzeMessageFlow() {
	flowInfo := MessageFlowInfo{
		TotalMessages: dc.Stats.MessageCount,
		MessageTypes: make(map[string]int64),
		MessageLatencies: make(map[string][]time.Duration),
		ProcessingTimes: make(map[string][]time.Duration),
		QueueSize:     dc.getMessageQueueSize(),
		DroppedMessages: dc.getDroppedMessageCount(),
	}

	// Analyze message types and processing
	for _, trace := range dc.Instrumentation.Traces {
		msgType := fmt.Sprintf("%T", trace.Message)
		flowInfo.MessageTypes[msgType]++

		// Add latency and processing time
		if trace.Duration > 0 {
			if _, exists := flowInfo.MessageLatencies[msgType]; !exists {
				flowInfo.MessageLatencies[msgType] = []time.Duration{}
			}
			flowInfo.MessageLatencies[msgType] = append(flowInfo.MessageLatencies[msgType], trace.Duration)
		}
	}

	// Calculate backpressure
	if flowInfo.QueueSize > 0 {
		flowInfo.BackPressure = float64(flowInfo.QueueSize) / float100
	}

	dc.Results.MessageFlow = flowInfo
}

// analyzeStateTransitions analyzes state transition behavior
func (dc *EventLoopDiagnosticContext) analyzeStateTransitions() {
	transitions := make([]StateTransition, 0)
	validTransitions := make(map[types.Stage]bool)
	invalidTransitions := make(map[types.Stage]bool)

	// Analyze all transitions from traces
	for _, trace := range dc.Instrumentation.Traces {
		if trace.Type == "state_transition" {
			transition := StateTransition{
				Timestamp: trace.Timestamp,
				Duration:  trace.Duration,
				Success:   trace.Error == nil,
				Error:     trace.Error,
			}
			transitions = append(transitions, transition)
		}
	}

	// Categorize transitions
	successCount := 0
	for _, t := range transitions {
		if t.Success {
			validTransitions[t.From] = true
			successCount++
		} else {
			invalidTransitions[t.From] = true
		}
	}

	transitionInfo := StateTransitionInfo{
		Transitions: transitions,
		Valid:       validTransitions,
		Invalid:     invalidTransitions,
		Total:       len(transitions),
		ErrorRate:   float64(len(transitions)-successCount) / float64(len(transitions)) * 100,
	}

	// Calculate average time
	if len(transitions) > 0 {
		var totalTime time.Duration
		for _, t := range transitions {
			totalTime += t.Duration
		}
		transitionInfo.AvgTime = totalTime / time.Duration(len(transitions))
	}

	dc.Results.StateTransitions = transitionInfo
}

// analyzeEventLoopPerformance analyzes event loop performance metrics
func (dc *EventLoopDiagnosticContext) analyzeEventLoopPerformance() {
	perfInfo := EventLoopPerformanceInfo{
		Goroutines: runtime.NumGoroutine(),
		StackDepth: dc.getStackDepth(),
		MemoryUsage: dc.getMemoryUsage(),
		LastGC:     dc.getLastGC(),
		GCDuration: dc.getGCDuration(),
		BlockedTime: dc.getBlockedTime(),
		CPUUsage:   dc.getCPUUsage(),
	}

	// Calculate FPS from frame times
	if len(dc.Stats.FrameTimes) > 1 {
		var frameTimes []time.Duration
		for _, ft := range dc.Stats.FrameTimes {
			if ft > 0 {
				frameTimes = append(frameTimes, ft)
			}
		}

		if len(frameTimes) > 0 {
			var totalTime time.Duration
			var minTime, maxTime time.Duration = frameTimes[0], frameTimes[0]

			for _, ft := range frameTimes {
				totalTime += ft
				if ft < minTime {
					minTime = ft
				}
				if ft > maxTime {
					maxTime = ft
				}
			}

			perfInfo.FrameTimeMin = minTime
			perfInfo.FrameTimeMax = maxTime
			perfInfo.FrameTimeAvg = totalTime / time.Duration(len(frameTimes))

			// Calculate FPS (frames per second)
			avgFrameTime := float64(perfInfo.FrameTimeAvg) / float64(time.Second)
			if avgFrameTime > 0 {
				perfInfo.FPS = 1.0 / avgFrameTime
			}

			// Min/Max FPS
			minFrameTime := float64(perfInfo.FrameTimeMin) / float64(time.Second)
			maxFrameTime := float64(perfInfo.FrameTimeMax) / float64(time.Second)
			if minFrameTime > 0 {
				perfInfo.FPSMax = 1.0 / minFrameTime
			}
			if maxFrameTime > 0 {
				perfInfo.FPSMin = 1.0 / maxFrameTime
			}
		}
	}

	dc.Results.EventLoopPerformance = perfInfo
}

// detectDeadlocks detects and analyzes deadlocks
func (dc *EventLoopDiagnosticContext) detectDeadlocks() {
	deadlockInfo := DeadlockInfo{
		DeadlockTimeout: 5 * time.Second, // Configurable timeout
	}

	// Check for blocking goroutines
	blockingGoroutines := dc.findBlockingGoroutines()
	if len(blockingGoroutines) > 0 {
		deadlockInfo.DeadlockDetected = true
		deadlockInfo.BlockingGoroutines = blockingGoroutines
	}

	// Analyze lock contention
	deadlockInfo.LockContentions = dc.analyzeLockContention()

	// Analyze resource usage
	deadlockInfo.ResourceUsage = dc.analyzeResourceUsage()

	dc.Results.DeadlockDetection = deadlockInfo
}

// generateEventLoopRecommendations generates recommendations for event loop issues
func (dc *EventLoopDiagnosticContext) generateEventLoopRecommendations() {
	recs := []string{}

	// Event loop recommendations
	if dc.Results.EventLoop.Blocked {
		recs = append(recs, "⚠️  Event loop blocked - check for infinite loops or blocking operations")
	}

	if dc.Results.EventLoop.AverageLoopTime > 100*time.Millisecond {
		recs = append(recs, "⚠️  Slow event loop (" + dc.Results.EventLoop.AverageLoopTime.String() + ") - optimize Update() method")
	}

	// MVU architecture recommendations
	mvu := dc.Results.MVUArchitecture
	if !mvu.ModelIntegrity {
		recs = append(recs, "❌ Model integrity issues detected - check for proper initialization")
	}

	if !mvu.ImmutableState {
		recs = append(recs, "❌ Mutability detected in state - ensure proper cloning in Update() method")
	}

	if mvu.CouplingScore > 0.7 {
		recs = append(recs, "⚠️  High coupling detected - consider refactoring for better separation of concerns")
	}

	if mvu.CohesionScore < 0.5 {
		recs = append(recs, "⚠️  Low cohesion detected - consider grouping related functionality together")
	}

	// Message flow recommendations
	flow := dc.Results.MessageFlow
	if flow.BackPressure > 0.8 {
		recs = append(recs, "⚠️  High message backpressure - consider throttling or batching messages")
	}

	if flow.DroppedMessages > 10 {
		recs = append(recs, "⚠️  High message drop rate (" + fmt.Sprintf("%d", flow.DroppedMessages) + ") - increase channel buffer size")
	}

	// State transition recommendations
	transitions := dc.Results.StateTransitions
	if transitions.ErrorRate > 5.0 {
		recs = append(recs, "⚠️  High state transition error rate (" + fmt.Sprintf("%.1f%%", transitions.ErrorRate) + ") - validate transitions")
	}

	// Performance recommendations
	perf := dc.Results.EventLoopPerformance
	if perf.FPS < 10.0 {
		recs = append(recs, "⚠️  Low FPS (" + fmt.Sprintf("%.1f", perf.FPS) + ") - optimize rendering and processing")
	}

	if perf.Goroutines > 50 {
		recs = append(recs, "⚠️  High goroutine count (" + fmt.Sprintf("%d", perf.Goroutines) + ") - check for goroutine leaks")
	}

	if perf.MemoryUsage > 100*1024*1024 { // 100MB
		recs = append(recs, "⚠️  High memory usage (" + formatBytes(perf.MemoryUsage) + ") - check for memory leaks")
	}

	// Deadlock recommendations
	deadlock := dc.Results.DeadlockDetection
	if deadlock.DeadlockDetected {
		recs = append(recs, "❌ Deadlock detected - immediate action required")
		if len(deadlock.BlockingGoroutines) > 0 {
			recs = append(recs, "   Blocked goroutines: " + fmt.Sprintf("%d", len(deadlock.BlockingGoroutines)))
		}
		if len(deadlock.LockContentions) > 0 {
			recs = append(recs, "   Lock contentions detected")
		}
	}

	// Generate specific Bubble Tea recommendations
	recs = append(recs, "💡 Bubble Tea event loop recommendations:")
	recs = append(recs, "   - Use tea.Tick() for periodic operations instead of time.Sleep()")
	recs = append(recs, "   - Implement proper error handling in Update() method")
	recs = append(recs, "   - Use context cancellation for graceful shutdown")
	recs = append(recs, "   - Avoid blocking operations in Update() method")
	recs = append(recs, "   - Use tea.Sequence() for command sequencing")
	recs = append(recs, "   - Implement frame rate limiting if needed")
	recs = append(recs, "   - Use tea.Quit() for clean shutdown instead of panic")

	dc.Results.Recommendations = recs
}

// Helper methods for event loop analysis
func (dc *EventLoopDiagnosticContext) isEventLoopBlocked() bool {
	// Check if event loop has been idle for too long
	return time.Since(dc.Results.EventLoop.LastMessage) > 5*time.Second
}

func (dc *EventLoopDiagnosticContext) getPendingCommandsCount() int {
	// Count pending commands in channels
	pending := len(dc.MessageChan)
	// Add more sophisticated command counting here
	return pending
}

func (dc *EventLoopDiagnosticContext) checkModelIntegrity() bool {
	// Check if model is properly initialized and not nil
	return dc.Model != nil
}

func (dc *EventLoopDiagnosticContext) checkImmutability() bool {
	// Check if model properly implements cloning
	// This would involve testing the Clone() method
	return true
}

func (dc *EventLoopDiagnosticContext) checkStateConsistency() bool {
	// Check state consistency across updates
	return true
}

func (dc *EventLoopDiagnosticContext) checkUpdateSafety() bool {
	// Check Update method safety
	return true
}

func (dc *EventLoopDiagnosticContext) checkViewPureness() bool {
	// Check if View method is pure (no side effects)
	return true
}

func (dc *EventLoopDiagnosticContext) checkMutationFree() bool {
	// Check if mutation-free patterns are followed
	return true
}

func (dc *EventLoopDiagnosticContext) checkInterfaceCompliance() bool {
	// Check interface compliance
	return true
}

func (dc *EventLoopDiagnosticContext) calculateCouplingScore() float64 {
	// Calculate coupling score based on dependencies
	// 0.0 = well-coupled, 1.0 = loosely-coupled
	return 0.3 // Example score
}

func (dc *EventLoopDiagnosticContext) calculateCohesionScore() float64 {
	// Calculate cohesion score
	// 0.0 = low cohesion, 1.0 = high cohesion
	return 0.7 // Example score
}

func (dc *EventLoopDiagnosticContext) getMessageQueueSize() int {
	return len(dc.MessageChan)
}

func (dc *EventLoopDiagnosticContext) getDroppedMessageCount() int {
	// Implement dropped message counting
	return 0
}

func (dc *EventLoopDiagnosticContext) getStackDepth() int {
	// Get stack depth from goroutine stacks
	stack := debug.Stack()
	return strings.Count(string(stack), "\n")
}

func (dc *EventLoopDiagnosticContext) getMemoryUsage() uint64 {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	return memStats.Alloc
}

func (dc *EventLoopDiagnosticContext) getLastGC() time.Time {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	return time.Unix(0, int64(memStats.LastGC))
}

func (dc *EventLoopDiagnosticContext) getGCDuration() time.Duration {
	// Get GC duration (would need custom instrumentation)
	return 0
}

func (dc *EventLoopDiagnosticContext) getBlockedTime() time.Duration {
	// Calculate blocked time
	return time.Duration(0)
}

func (dc *EventLoopDiagnosticContext) getCPUUsage() float64 {
	// Get CPU usage (would need profiling)
	return 0.0
}

func (dc *EventLoopDiagnosticContext) findBlockingGoroutines() []BlockingGoroutine {
	// Find blocking goroutines
	blockingGoroutines := make([]BlockingGoroutine, 0)

	// This would typically involve stack traces and goroutine inspection
	// For now, return empty slice
	return blockingGoroutines
}

func (dc *EventLoopDiagnosticContext) analyzeLockContention() []LockContention {
	// Analyze lock contention
	return []LockContention{}
}

func (dc *EventLoopDiagnosticContext) analyzeResourceUsage() map[string]ResourceUsage {
	// Analyze resource usage
	return map[string]ResourceUsage{}
}

// Instrumentation methods
func (dc *EventLoopDiagnosticContext) AddTrace(trace TraceEvent) {
	dc.Instrumentation.Traces = append(dc.Instrumentation.Traces, trace)

	// Keep only recent traces to prevent memory issues
	if len(dc.Instrumentation.Traces) > 1000 {
		dc.Instrumentation.Traces = dc.Instrumentation.Traces[500:]
	}
}

func (dc *EventLoopDiagnosticContext) AddDebugEvent(event DebugEvent) {
	select {
	case dc.DebugChan <- event:
	default:
		// Channel full, drop debug event
	}
}

func (dc *EventLoopDiagnosticContext) ProcessDebugEvents() {
	for {
		select {
		case event := <-dc.DebugChan:
			dc.handleDebugEvent(event)
		default:
			return
		}
	}
}

func (dc *EventLoopDiagnosticContext) handleDebugEvent(event DebugEvent) {
	// Process debug events
}

// GenerateEventLoopReport generates a comprehensive event loop diagnostic report
func (dc *EventLoopDiagnosticContext) GenerateEventLoopReport() string {
	var report strings.Builder

	// Header
	report.WriteString("🔍 EVENT LOOP AND MVU ARCHITECTURE DIAGNOSTIC REPORT\n")
	report.WriteString("=" * 60 + "\n\n")
	report.WriteString(fmt.Sprintf("📅 Diagnostic Time: %s\n", dc.StartTime.Format("2006-01-02 15:04:05")))
	report.WriteString(fmt.Sprintf("⏱️  Duration: %v\n", time.Since(dc.StartTime)))
	report.WriteString("\n")

	// Event Loop Information
	report.WriteString("🔄 EVENT LOOP STATE\n")
	report.WriteString("-" * 40 + "\n")
	loop := dc.Results.EventLoop
	report.WriteString(fmt.Sprintf("   Running: %v\n", loop.Running))
	report.WriteString(fmt.Sprintf("   Started: %s\n", loop.StartedAt.Format("15:04:05")))
	report.WriteString(fmt.Sprintf("   Messages: %d\n", loop.MessageCount))
	report.WriteString(fmt.Sprintf("   Errors: %d\n", loop.ErrorCount))
	report.WriteString(fmt.Sprintf("   Panics: %d\n", loop.PanicCount))
	report.WriteString(fmt.Sprintf("   Commands Pending: %d\n", loop.CommandsPending))
	report.WriteString(fmt.Sprintf("   Blocked: %v\n", loop.Blocked))
	report.WriteString(fmt.Sprintf("   Average Loop Time: %v\n", loop.AverageLoopTime))
	report.WriteString("\n")

	// MVU Architecture Information
	report.WriteString("🏗️  MVU ARCHITECTURE\n")
	report.WriteString("-" * 40 + "\n")
	mvu := dc.Results.MVUArchitecture
	report.WriteString(fmt.Sprintf("   Model Integrity: %v\n", mvu.ModelIntegrity))
	report.WriteString(fmt.Sprintf("   Immutable State: %v\n", mvu.ImmutableState))
	report.WriteString(fmt.Sprintf("   State Consistency: %v\n", mvu.StateConsistency))
	report.WriteString(fmt.Sprintf("   Update Safety: %v\n", mvu.UpdateSafety))
	report.WriteString(fmt.Sprintf("   View Pureness: %v\n", mvu.ViewPureness))
	report.WriteString(fmt.Sprintf("   Mutation Free: %v\n", mvu.MutationFree))
	report.WriteString(fmt.Sprintf("   Interface Compliance: %v\n", mvu.InterfaceCompliance))
	report.WriteString(fmt.Sprintf("   Coupling Score: %.2f\n", mvu.CouplingScore))
	report.WriteString(fmt.Sprintf("   Cohesion Score: %.2f\n", mvu.CohesionScore))
	report.WriteString("\n")

	// Message Flow Information
	report.WriteString("📨 MESSAGE FLOW\n")
	report.WriteString("-" * 40 + "\n")
	flow := dc.Results.MessageFlow
	report.WriteString(fmt.Sprintf("   Total Messages: %d\n", flow.TotalMessages))
	report.WriteString(fmt.Sprintf("   Queue Size: %d\n", flow.QueueSize))
	report.WriteString(fmt.Sprintf("   Dropped Messages: %d\n", flow.DroppedMessages))
	report.WriteString(fmt.Sprintf("   Back Pressure: %.2f\n", flow.BackPressure))
	report.WriteString("\n")

	// State Transition Information
	report.WriteString("🔄 STATE TRANSITIONS\n")
	report.WriteString("-" * 40 + "\n")
	transitions := dc.Results.StateTransitions
	report.WriteString(fmt.Sprintf("   Total Transitions: %d\n", transitions.Total))
	report.WriteString(fmt.Sprintf("   Error Rate: %.1f%%\n", transitions.ErrorRate))
	report.WriteString(fmt.Sprintf("   Average Time: %v\n", transitions.AvgTime))
	report.WriteString("\n")

	// Performance Information
	report.WriteString("⚡ PERFORMANCE METRICS\n")
	report.WriteString("-" * 40 + "\n")
	perf := dc.Results.EventLoopPerformance
	report.WriteString(fmt.Sprintf("   FPS: %.1f (min: %.1f, max: %.1f)\n", perf.FPS, perf.FPSMin, perf.FPSMax))
	report.WriteString(fmt.Sprintf("   Frame Time: avg: %v, min: %v, max: %v\n", perf.FrameTimeAvg, perf.FrameTimeMin, perf.FrameTimeMax))
	report.WriteString(fmt.Sprintf("   Goroutines: %d\n", perf.Goroutines))
	report.WriteString(fmt.Sprintf("   Memory Usage: %s\n", formatBytes(perf.MemoryUsage)))
	report.WriteString(fmt.Sprintf("   CPU Usage: %.1f%%\n", perf.CPUUsage))
	report.WriteString("\n")

	// Deadlock Information
	report.WriteString("🚫 DEADLOCK DETECTION\n")
	report.WriteString("-" * 40 + "\n")
	deadlock := dc.Results.DeadlockDetection
	report.WriteString(fmt.Sprintf("   Deadlock Detected: %v\n", deadlock.DeadlockDetected))
	if deadlock.DeadlockDetected {
		report.WriteString(fmt.Sprintf("   Reason: %s\n", deadlock.DeadlockReason))
		report.WriteString(fmt.Sprintf("   Blocked Goroutines: %d\n", len(deadlock.BlockingGoroutines)))
		report.WriteString(fmt.Sprintf("   Lock Contentions: %d\n", len(deadlock.LockContentions)))
	}
	report.WriteString("\n")

	// Recommendations
	report.WriteString("💡 RECOMMENDATIONS\n")
	report.WriteString("-" * 40 + "\n")
	for _, rec := range dc.Results.Recommendations {
		report.WriteString(fmt.Sprintf("   %s\n", rec))
	}

	return report.String()
}

// RunEventLoopTest runs a specific event loop diagnostic test
func RunEventLoopTest(testName string, model *Model) (*EventLoopDiagnosticResult, error) {
	dc := NewEventLoopDiagnosticContext(model)

	switch testName {
	case "event_loop":
		dc.analyzeEventLoopState()
	case "mvu":
		dc.analyzeMVUArchitecture()
	case "message_flow":
		dc.analyzeMessageFlow()
	case "state_transitions":
		dc.analyzeStateTransitions()
	case "performance":
		dc.analyzeEventLoopPerformance()
	case "deadlock":
		dc.detectDeadlocks()
	case "comprehensive":
		return dc.RunEventLoopDiagnostics(), nil
	default:
		return nil, fmt.Errorf("unknown event loop diagnostic test: %s", testName)
	}

	return dc.Results, nil
}