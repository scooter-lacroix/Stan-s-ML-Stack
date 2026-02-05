// internal/ui/performance/monitor.go
package performance

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/scooter-lacroix/mlstack-installer/internal/ui/types"
)

// PerformanceMonitor implements comprehensive performance monitoring
type PerformanceMonitor struct {
	// Metrics tracking
	metrics          map[string]*Metric
	operationMetrics map[string]*OperationMetrics
	systemMetrics    *SystemMetrics

	// Configuration
	maxMetrics          int
	metricRetention     time.Duration
	enableSystemMetrics bool

	// Performance thresholds
	warningThresholds map[string]time.Duration
	errorThresholds   map[string]time.Duration

	// Concurrency safety
	mu         sync.RWMutex
	startTime  time.Time
	frameCount int64

	// Alerting
	alertCallbacks []AlertCallback
	enabled        bool
}

// Metric represents a performance metric
type Metric struct {
	Name      string
	Value     float64
	Timestamp time.Time
	Tags      map[string]string
}

// OperationMetrics tracks metrics for specific operations
type OperationMetrics struct {
	Name         string
	StartTime    time.Time
	EndTime      time.Time
	Duration     time.Duration
	Success      bool
	Error        error
	AttemptCount int
	SubMetrics   map[string]*Metric
}

// SystemMetrics tracks system-wide performance metrics
type SystemMetrics struct {
	CPUUsage       float64
	MemoryUsage    float64
	DiskUsage      float64
	GPUUsage       []float64
	ThreadCount    int
	GoroutineCount int
	Uptime         time.Duration
	LastUpdate     time.Time
}

// AlertCallback is a function that gets called when performance alerts are triggered
type AlertCallback func(alert types.PerformanceAlert)

// PerformanceAlert represents a performance alert (moved to types package)

// NewMonitor creates a new performance monitor
func NewMonitor() *PerformanceMonitor {
	monitor := &PerformanceMonitor{
		metrics:             make(map[string]*Metric),
		operationMetrics:    make(map[string]*OperationMetrics),
		systemMetrics:       &SystemMetrics{},
		maxMetrics:          10000,
		metricRetention:     24 * time.Hour,
		enableSystemMetrics: true,
		warningThresholds:   make(map[string]time.Duration),
		errorThresholds:     make(map[string]time.Duration),
		startTime:           time.Now(),
		alertCallbacks:      make([]AlertCallback, 0),
		enabled:             true,
	}

	// Set default thresholds
	monitor.SetDefaultThresholds()

	// Start system metrics collection
	go monitor.collectSystemMetrics()

	return monitor
}

// SetDefaultThresholds sets default performance thresholds
func (pm *PerformanceMonitor) SetDefaultThresholds() {
	pm.warningThresholds["frame_time"] = 100 * time.Millisecond
	pm.warningThresholds["operation_duration"] = 5 * time.Second
	pm.warningThresholds["response_time"] = 1 * time.Second

	pm.errorThresholds["frame_time"] = 500 * time.Millisecond
	pm.errorThresholds["operation_duration"] = 30 * time.Second
	pm.errorThresholds["response_time"] = 10 * time.Second
}

// StartOperation starts tracking an operation
func (pm *PerformanceMonitor) StartOperation(name string) {
	if !pm.enabled {
		return
	}

	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Remove existing operation if any
	delete(pm.operationMetrics, name)

	// Start new operation
	operation := &OperationMetrics{
		Name:       name,
		StartTime:  time.Now(),
		SubMetrics: make(map[string]*Metric),
	}
	pm.operationMetrics[name] = operation

	pm.RecordMetric("operation_start", float64(time.Since(pm.startTime).Milliseconds()))
}

// FinishOperation ends tracking an operation
func (pm *PerformanceMonitor) FinishOperation(name string, success bool) {
	if !pm.enabled {
		return
	}

	pm.mu.Lock()
	defer pm.mu.Unlock()

	if operation, exists := pm.operationMetrics[name]; exists {
		operation.EndTime = time.Now()
		operation.Duration = operation.EndTime.Sub(operation.StartTime)
		operation.Success = success

		// Record operation metrics
		pm.RecordMetric("operation_duration", float64(operation.Duration.Milliseconds()))
		pm.RecordMetric("operation_success", float64(btoi(success)))

		// Check thresholds
		pm.checkOperationThresholds(operation)

		// Clean up
		delete(pm.operationMetrics, name)
	}
}

// RecordMetric records a performance metric
func (pm *PerformanceMonitor) RecordMetric(metricName string, value float64) {
	if !pm.enabled {
		return
	}

	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Create metric
	metric := &Metric{
		Name:      metricName,
		Value:     value,
		Timestamp: time.Now(),
		Tags:      make(map[string]string),
	}

	// Add to metrics
	pm.metrics[metricName] = metric

	// Limit metrics count
	if len(pm.metrics) > pm.maxMetrics {
		pm.cleanupOldMetrics()
	}

	// Check thresholds
	pm.checkMetricThresholds(metricName, value)

	// Update frame count
	atomic.AddInt64(&pm.frameCount, 1)
}

// RecordMetricWithTags records a performance metric with tags
func (pm *PerformanceMonitor) RecordMetricWithTags(metricName string, value float64, tags map[string]string) {
	if !pm.enabled {
		return
	}

	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Create metric with tags
	metric := &Metric{
		Name:      metricName,
		Value:     value,
		Timestamp: time.Now(),
		Tags:      tags,
	}

	// Add to metrics
	pm.metrics[metricName] = metric

	// Limit metrics count
	if len(pm.metrics) > pm.maxMetrics {
		pm.cleanupOldMetrics()
	}

	// Check thresholds
	pm.checkMetricThresholds(metricName, value)

	// Update frame count
	atomic.AddInt64(&pm.frameCount, 1)
}

// GetMetrics returns all recorded metrics
func (pm *PerformanceMonitor) GetMetrics() map[string]float64 {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	result := make(map[string]float64)
	for name, metric := range pm.metrics {
		result[name] = metric.Value
	}
	return result
}

// GetOperationMetrics returns metrics for specific operations
func (pm *PerformanceMonitor) GetOperationMetrics() map[string]*OperationMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Create a copy to avoid race conditions
	result := make(map[string]*OperationMetrics)
	for name, metrics := range pm.operationMetrics {
		copy := *metrics
		result[name] = &copy
	}
	return result
}

// GetSystemMetrics returns system-wide performance metrics
func (pm *PerformanceMonitor) GetSystemMetrics() *SystemMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Create a copy to avoid race conditions
	copy := *pm.systemMetrics
	return &copy
}

// GetMetricsHistory returns metrics history for a specific metric name
func (pm *PerformanceMonitor) GetMetricsHistory(metricName string, duration time.Duration) []*Metric {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	cutoff := time.Now().Add(-duration)
	var history []*Metric

	for _, metric := range pm.metrics {
		if metric.Name == metricName && metric.Timestamp.After(cutoff) {
			history = append(history, metric)
		}
	}

	return history
}

// GetOperationHistory returns operation history
func (pm *PerformanceMonitor) GetOperationHistory(duration time.Duration) []*OperationMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	cutoff := time.Now().Add(-duration)
	var history []*OperationMetrics

	for _, operation := range pm.operationMetrics {
		if operation.StartTime.After(cutoff) {
			history = append(history, operation)
		}
	}

	return history
}

// GetPerformanceReport generates a comprehensive performance report
func (pm *PerformanceMonitor) GetPerformanceReport() map[string]interface{} {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	report := map[string]interface{}{
		"uptime":            time.Since(pm.startTime),
		"frame_count":       atomic.LoadInt64(&pm.frameCount),
		"current_metrics":   len(pm.metrics),
		"active_operations": len(pm.operationMetrics),
		"enabled":           pm.enabled,
		"thresholds": map[string]interface{}{
			"warning": pm.warningThresholds,
			"error":   pm.errorThresholds,
		},
		"system_metrics": pm.systemMetrics,
		"metrics":        pm.getMetricsSummary(),
		"operations":     pm.getOperationsSummary(),
	}

	return report
}

// getMetricsSummary returns a summary of current metrics
func (pm *PerformanceMonitor) getMetricsSummary() map[string]interface{} {
	summary := make(map[string]interface{})

	// Count metrics by type
	metricTypes := make(map[string]int)
	totalValue := 0.0

	for name, metric := range pm.metrics {
		metricTypes[name]++
		totalValue += metric.Value
	}

	summary["count"] = len(pm.metrics)
	summary["types"] = metricTypes
	summary["average_value"] = totalValue / float64(len(pm.metrics))

	return summary
}

// getOperationsSummary returns a summary of operations
func (pm *PerformanceMonitor) getOperationsSummary() map[string]interface{} {
	summary := make(map[string]interface{})

	// Count operations by type
	operationTypes := make(map[string]int)
	successCount := 0
	totalDuration := time.Duration(0)

	for name, operation := range pm.operationMetrics {
		operationTypes[name]++
		if operation.Success {
			successCount++
		}
		if operation.EndTime.After(operation.StartTime) {
			totalDuration += operation.Duration
		}
	}

	summary["count"] = len(pm.operationMetrics)
	summary["types"] = operationTypes
	summary["success_rate"] = float64(successCount) / float64(len(pm.operationMetrics))
	summary["average_duration"] = float64(totalDuration.Milliseconds()) / float64(len(pm.operationMetrics))

	return summary
}

// SetThreshold sets a performance threshold
func (pm *PerformanceMonitor) SetThreshold(metricName string, threshold time.Duration, severity string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	switch severity {
	case "warning":
		pm.warningThresholds[metricName] = threshold
	case "error":
		pm.errorThresholds[metricName] = threshold
	default:
		panic(fmt.Sprintf("invalid severity level: %s", severity))
	}
}

// checkMetricThresholds checks if metrics exceed thresholds
func (pm *PerformanceMonitor) checkMetricThresholds(metricName string, value float64) {
	// Check warning threshold
	if warningThreshold, exists := pm.warningThresholds[metricName]; exists {
		if time.Duration(value*float64(time.Millisecond)) > warningThreshold {
			pm.triggerAlert(types.PerformanceAlert{
				Type:       "metric_warning",
				Message:    fmt.Sprintf("Metric %s exceeded warning threshold", metricName),
				MetricName: metricName,
				Value:      value,
				Threshold:  float64(warningThreshold.Milliseconds()),
				Timestamp:  time.Now(),
				Severity:   "warning",
			})
		}
	}

	// Check error threshold
	if errorThreshold, exists := pm.errorThresholds[metricName]; exists {
		if time.Duration(value*float64(time.Millisecond)) > errorThreshold {
			pm.triggerAlert(types.PerformanceAlert{
				Type:       "metric_error",
				Message:    fmt.Sprintf("Metric %s exceeded error threshold", metricName),
				MetricName: metricName,
				Value:      value,
				Threshold:  float64(errorThreshold.Milliseconds()),
				Timestamp:  time.Now(),
				Severity:   "error",
			})
		}
	}
}

// checkOperationThresholds checks if operations exceed thresholds
func (pm *PerformanceMonitor) checkOperationThresholds(operation *OperationMetrics) {
	operationDuration := float64(operation.Duration.Milliseconds())

	// Check warning threshold
	if warningThreshold, exists := pm.warningThresholds["operation_duration"]; exists {
		if operation.Duration > warningThreshold {
			pm.triggerAlert(types.PerformanceAlert{
				Type:       "operation_warning",
				Message:    fmt.Sprintf("Operation %s exceeded warning threshold", operation.Name),
				MetricName: operation.Name,
				Value:      operationDuration,
				Threshold:  float64(warningThreshold.Milliseconds()),
				Timestamp:  time.Now(),
				Severity:   "warning",
			})
		}
	}

	// Check error threshold
	if errorThreshold, exists := pm.errorThresholds["operation_duration"]; exists {
		if operation.Duration > errorThreshold {
			pm.triggerAlert(types.PerformanceAlert{
				Type:       "operation_error",
				Message:    fmt.Sprintf("Operation %s exceeded error threshold", operation.Name),
				MetricName: operation.Name,
				Value:      operationDuration,
				Threshold:  float64(errorThreshold.Milliseconds()),
				Timestamp:  time.Now(),
				Severity:   "error",
			})
		}
	}
}

// triggerAlert triggers performance alerts
func (pm *PerformanceMonitor) triggerAlert(alert types.PerformanceAlert) {
	for _, callback := range pm.alertCallbacks {
		callback(alert)
	}
}

// AddAlertCallback adds an alert callback function
func (pm *PerformanceMonitor) AddAlertCallback(callback AlertCallback) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.alertCallbacks = append(pm.alertCallbacks, callback)
}

// EnableSystemMetrics enables or disables system metrics collection
func (pm *PerformanceMonitor) EnableSystemMetrics(enabled bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.enableSystemMetrics = enabled
}

// SetMaxMetrics sets the maximum number of metrics to retain
func (pm *PerformanceMonitor) SetMaxMetrics(max int) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.maxMetrics = max
}

// SetMetricRetention sets the retention period for metrics
func (pm *PerformanceMonitor) SetMetricRetention(duration time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.metricRetention = duration
}

// cleanupOldMetrics removes old metrics to prevent memory issues
func (pm *PerformanceMonitor) cleanupOldMetrics() {
	cutoff := time.Now().Add(-pm.metricRetention)

	for name, metric := range pm.metrics {
		if metric.Timestamp.Before(cutoff) {
			delete(pm.metrics, name)
		}
	}
}

// collectSystemMetrics collects system-wide performance metrics
func (pm *PerformanceMonitor) collectSystemMetrics() {
	if !pm.enabled || !pm.enableSystemMetrics {
		return
	}

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		pm.updateSystemMetrics()
	}
}

// updateSystemMetrics updates system performance metrics
func (pm *PerformanceMonitor) updateSystemMetrics() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Get system metrics
	cpuUsage := pm.getCPUUsage()
	memoryUsage := pm.getMemoryUsage()
	goroutineCount := runtime.NumGoroutine()

	// Update system metrics
	pm.systemMetrics = &SystemMetrics{
		CPUUsage:       cpuUsage,
		MemoryUsage:    memoryUsage,
		DiskUsage:      pm.getDiskUsage(),
		GPUUsage:       pm.getGPUUsage(),
		ThreadCount:    runtime.NumCPU(),
		GoroutineCount: goroutineCount,
		Uptime:         time.Since(pm.startTime),
		LastUpdate:     time.Now(),
	}

	// Record system metrics
	pm.RecordMetric("cpu_usage_percent", cpuUsage)
	pm.RecordMetric("memory_usage_percent", memoryUsage)
	pm.RecordMetric("goroutine_count", float64(goroutineCount))

	// Check system resource thresholds
	pm.checkSystemResourceThresholds()
}

// getCPUUsage gets current CPU usage percentage
func (pm *PerformanceMonitor) getCPUUsage() float64 {
	// This is a simplified implementation
	// In a real implementation, you would use proper system monitoring
	return 25.0 // Default value for demo
}

// getMemoryUsage gets current memory usage percentage
func (pm *PerformanceMonitor) getMemoryUsage() float64 {
	// This is a simplified implementation
	// In a real implementation, you would use proper system monitoring
	return 40.0 // Default value for demo
}

// getDiskUsage gets current disk usage percentage
func (pm *PerformanceMonitor) getDiskUsage() float64 {
	// This is a simplified implementation
	// In a real implementation, you would use proper system monitoring
	return 60.0 // Default value for demo
}

// getGPUUsage gets current GPU usage percentages
func (pm *PerformanceMonitor) getGPUUsage() []float64 {
	// This is a simplified implementation
	// In a real implementation, you would use proper GPU monitoring
	return []float64{45.0, 30.0} // Default values for demo
}

// checkSystemResourceThresholds checks system resource usage against thresholds
func (pm *PerformanceMonitor) checkSystemResourceThresholds() {
	// Check CPU usage
	if pm.systemMetrics.CPUUsage > 80.0 {
		pm.triggerAlert(types.PerformanceAlert{
			Type:      "system_warning",
			Message:   "High CPU usage detected",
			Value:     pm.systemMetrics.CPUUsage,
			Threshold: 80.0,
			Timestamp: time.Now(),
			Severity:  "warning",
		})
	}

	// Check memory usage
	if pm.systemMetrics.MemoryUsage > 80.0 {
		pm.triggerAlert(types.PerformanceAlert{
			Type:      "system_warning",
			Message:   "High memory usage detected",
			Value:     pm.systemMetrics.MemoryUsage,
			Threshold: 80.0,
			Timestamp: time.Now(),
			Severity:  "warning",
		})
	}

	// Check goroutine count
	if float64(pm.systemMetrics.GoroutineCount) > 1000.0 {
		pm.triggerAlert(types.PerformanceAlert{
			Type:      "system_warning",
			Message:   "High goroutine count detected",
			Value:     float64(pm.systemMetrics.GoroutineCount),
			Threshold: 1000.0,
			Timestamp: time.Now(),
			Severity:  "warning",
		})
	}
}

// ResetMetrics resets all metrics
func (pm *PerformanceMonitor) ResetMetrics() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.metrics = make(map[string]*Metric)
	pm.operationMetrics = make(map[string]*OperationMetrics)
	pm.startTime = time.Now()
	atomic.StoreInt64(&pm.frameCount, 0)
}

// ExportMetrics exports metrics to a format suitable for analysis
func (pm *PerformanceMonitor) ExportMetrics() map[string]interface{} {
	return pm.GetPerformanceReport()
}

// GetTopMetrics returns the top N metrics by value
func (pm *PerformanceMonitor) GetTopMetrics(n int) []*Metric {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	metrics := make([]*Metric, 0, len(pm.metrics))
	for _, metric := range pm.metrics {
		metrics = append(metrics, metric)
	}

	// Sort by value (descending)
	for i := 0; i < len(metrics)-1; i++ {
		for j := i + 1; j < len(metrics); j++ {
			if metrics[i].Value < metrics[j].Value {
				metrics[i], metrics[j] = metrics[j], metrics[i]
			}
		}
	}

	if n > len(metrics) {
		n = len(metrics)
	}

	return metrics[:n]
}

// GetSlowestOperations returns the slowest operations
func (pm *PerformanceMonitor) GetSlowestOperations(n int) []*OperationMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	operations := make([]*OperationMetrics, 0, len(pm.operationMetrics))
	for _, operation := range pm.operationMetrics {
		operations = append(operations, operation)
	}

	// Sort by duration (descending)
	for i := 0; i < len(operations)-1; i++ {
		for j := i + 1; j < len(operations); j++ {
			if operations[i].Duration < operations[j].Duration {
				operations[i], operations[j] = operations[j], operations[i]
			}
		}
	}

	if n > len(operations) {
		n = len(operations)
	}

	return operations[:n]
}

// btoi converts bool to int
func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// SetEnabled enables or disables the performance monitor
func (pm *PerformanceMonitor) SetEnabled(enabled bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.enabled = enabled
}

// IsEnabled returns whether the performance monitor is enabled
func (pm *PerformanceMonitor) IsEnabled() bool {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return pm.enabled
}

// GetMetricsCount returns the number of current metrics
func (pm *PerformanceMonitor) GetMetricsCount() int {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return len(pm.metrics)
}

// GetOperationCount returns the number of active operations
func (pm *PerformanceMonitor) GetOperationCount() int {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return len(pm.operationMetrics)
}

// GetFrameCount returns the total number of frames processed
func (pm *PerformanceMonitor) GetFrameCount() int64 {
	return atomic.LoadInt64(&pm.frameCount)
}

// GetUptime returns the uptime of the performance monitor
func (pm *PerformanceMonitor) GetUptime() time.Duration {
	return time.Since(pm.startTime)
}
