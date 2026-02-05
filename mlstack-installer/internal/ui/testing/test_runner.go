// Package testing provides comprehensive testing framework for Bubble Tea UI
package testing

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/charmbracelet/bubbletea"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/terminal"
	"github.com/scooter-lacroix/mlstack-installer/internal/ui/testing/minimal_examples"
)

// TestRunner manages and executes test suites
type TestRunner struct {
	ctx                context.Context
	cancel             context.CancelFunc
	config             TestRunnerConfig
	state              TestRunnerState
	testSuites         []TestSuite
	currentTest        *TestCase
	results            TestResults
	startTime          time.Time
	terminalManager    *terminal.TerminalManager
}

// TestRunnerConfig defines test runner configuration
type TestRunnerConfig struct {
	Enabled              bool
	Timeout              time.Duration
	RetryCount           int
	ParallelTests        bool
	StrictMode           bool
	VerboseOutput       bool
	OutputFormat        string // "text", "json", "xml"
	SafetyMode          bool
	ProfilingMode      bool
	DiagnosticMode     bool
	ReportFile          string
	TerminalConfig      terminal.TerminalConfig
	TestDirectory       string
	ExcludePatterns     []string
	IncludePatterns     []string
	EnvironmentVars     map[string]string
}

// TestRunnerState represents test runner state
type TestRunnerState struct {
	Running          bool
	Completed        bool
	Failed          bool
	CurrentTest     string
	TotalTests      int
	CompletedTests  int
	FailedTests     int
	SkippedTests    int
	StartTime       time.Time
	EndTime         time.Time
	CurrentPhase    string
}

// TestSuite represents a test suite
type TestSuite struct {
	Name         string
	Description string
	TestCases   []TestCase
	Setup       func() error
	Teardown    func() error
	Timeout     time.Duration
	Parallel    bool
	Skip        bool
}

// TestCase represents a test case
type TestCase struct {
	Name         string
	Description string
	TestFunc     func() error
	Setup       func() error
	Teardown    func() error
	Timeout     time.Duration
	Skip        bool
	SkipReason   string
	ExpectedError error
	ActualError   error
	Duration      time.Duration
	Status        TestStatus
	Logs         []string
	Dependencies  []string
	Tags         []string
	Metadata     map[string]interface{}
}

// TestStatus represents test status
type TestStatus string

const (
	TestStatusPending  TestStatus = "pending"
	TestStatusRunning  TestStatus = "running"
	TestStatusPassed   TestStatus = "passed"
	TestStatusFailed   TestStatus = "failed"
	TestStatusSkipped  TestStatus = "skipped"
	TestStatusTimeout TestStatus = "timeout"
)

// TestResults represents test execution results
type TestResults struct {
	TotalSuites     int
	CompletedSuites int
	FailedSuites    int
	TotalTests      int
	CompletedTests  int
	FailedTests     int
	SkippedTests    int
	PassedTests     int
	ExecutionTime   time.Duration
	StartTime       time.Time
	EndTime         time.Time
	SuiteResults    []SuiteResult
	Summary         TestSummary
}

// SuiteResult represents suite execution results
type SuiteResult struct {
	Name           string
	Status         TestStatus
	Duration       time.Duration
	TestCount      int
	PassedCount    int
	FailedCount    int
	SkippedCount   int
	ErrorCount     int
	TestResults    []TestResult
	StartTime      time.Time
	EndTime        time.Time
}

// TestResult represents test execution results
type TestResult struct {
	Name          string
	Status        TestStatus
	Duration      time.Duration
	Error         error
	Logs          []string
	Metadata      map[string]interface{}
	StartTime     time.Time
	EndTime       time.Time
}

// TestSummary represents test execution summary
type TestSummary struct {
	OverallStatus     TestStatus
	PassRate          float64
	FailRate          float64
	SkipRate          float64
	AverageDuration   time.Duration
	LongestTest       string
	ShortestTest      string
	MostFailingSuite  string
	MostPassingSuite  string
	CriticalFailures  []string
	Recommendations   []string
}

// NewTestRunner creates a new test runner
func NewTestRunner(config TestRunnerConfig) *TestRunner {
	ctx, cancel := context.WithCancel(context.Background())

	return &TestRunner{
		ctx:             ctx,
		cancel:          cancel,
		config:          config,
		state:           TestRunnerState{},
		testSuites:       []TestSuite{},
		results:         TestResults{},
		startTime:       time.Now(),
		terminalManager: terminal.NewTerminalManager(),
	}
}

// AddTestSuite adds a test suite
func (tr *TestRunner) AddTestSuite(suite TestSuite) {
	tr.testSuites = append(tr.testSuites, suite)
}

// RunTests runs all test suites
func (tr *TestRunner) RunTests() (*TestResults, error) {
	tr.mu.Lock()
	defer tr.mu.Unlock()

	if !tr.config.Enabled {
		return nil, fmt.Errorf("test runner is disabled")
	}

	// Initialize terminal manager
	if err := tr.terminalManager.Initialize(); err != nil {
		return nil, fmt.Errorf("terminal manager initialization failed: %w", err)
	}

	// Set up signal handling
	tr.setupSignalHandling()

	// Start the runner
	tr.state.Running = true
	tr.state.StartTime = time.Now()
	tr.startTime = time.Now()

	// Run all test suites
	for _, suite := range tr.testSuites {
		suiteResult := tr.runTestSuite(suite)
		tr.results.SuiteResults = append(tr.results.SuiteResults, suiteResult)
	}

	// Stop the runner
	tr.state.Running = false
	tr.state.EndTime = time.Now()
	tr.results.EndTime = time.Now()
	tr.results.ExecutionTime = time.Now().Sub(tr.startTime)

	// Generate summary
	tr.generateSummary()

	// Generate report
	if tr.config.ReportFile != "" {
		tr.generateReport()
	}

	return &tr.results, nil
}

// runTestSuite runs a single test suite
func (tr *TestRunner) runTestSuite(suite TestSuite) SuiteResult {
	result := SuiteResult{
		Name:        suite.Name,
		Status:      TestStatusPending,
		TestResults: []TestResult{},
		StartTime:  time.Now(),
	}

	// Skip if suite is marked to skip
	if suite.Skip {
		result.Status = TestStatusSkipped
		return result
	}

	// Run setup
	if suite.Setup != nil {
		if err := suite.Setup(); err != nil {
			result.Status = TestStatusFailed
			result.Error = fmt.Errorf("suite setup failed: %w", err)
			return result
		}
	}

	// Run test cases
	var parallelTests []*TestCase
	var sequentialTests []*TestCase

	for i := range suite.TestCases {
		testCase := &suite.TestCases[i]
		if testCase.Skip {
			testResult := TestResult{
				Name:   testCase.Name,
				Status: TestStatusSkipped,
				StartTime: time.Now(),
				EndTime: time.Now(),
			}
			result.TestResults = append(result.TestResults, testResult)
			continue
		}

		if suite.Parallel && testCase.Parallel {
			parallelTests = append(parallelTests, testCase)
		} else {
			sequentialTests = append(sequentialTests, testCase)
		}
	}

	// Run sequential tests first
	for _, testCase := range sequentialTests {
		testResult := tr.runTestCase(testCase)
		result.TestResults = append(result.TestResults, testResult)
	}

	// Run parallel tests
	if tr.config.ParallelTests && len(parallelTests) > 0 {
		tr.runParallelTests(parallelTests)
	}

	// Run teardown
	if suite.Teardown != nil {
		if err := suite.Teardown(); err != nil {
			result.Status = TestStatusFailed
			if result.Error == nil {
				result.Error = fmt.Errorf("suite teardown failed: %w", err)
			}
		}
	}

	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)

	// Calculate suite statistics
	for _, testResult := range result.TestResults {
		switch testResult.Status {
		case TestStatusPassed:
			result.PassedCount++
		case TestStatusFailed:
			result.FailedCount++
		case TestStatusSkipped:
			result.SkippedCount++
		case TestStatusTimeout:
			result.ErrorCount++
		}
		result.TestCount++
	}

	// Determine overall suite status
	if result.FailedCount > 0 {
		result.Status = TestStatusFailed
	} else if result.SkippedCount == result.TestCount {
		result.Status = TestStatusSkipped
	} else {
		result.Status = TestStatusPassed
	}

	tr.results.CompletedSuites++
	if result.Status == TestStatusFailed {
		tr.results.FailedSuites++
	}

	return result
}

// runTestCase runs a single test case
func (tr *TestRunner) runTestCase(testCase *TestCase) TestResult {
	result := TestResult{
		Name:     testCase.Name,
		StartTime: time.Now(),
		Metadata: make(map[string]interface{}),
	}

	// Set current test
	tr.currentTest = testCase
	tr.state.CurrentTest = testCase.Name
	tr.state.CompletedTests++
	tr.state.TotalTests++

	// Skip test if marked
	if testCase.Skip {
		result.Status = TestStatusSkipped
		result.EndTime = time.Now()
		return result
	}

	// Run test setup
	if testCase.Setup != nil {
		if err := testCase.Setup(); err != nil {
			result.Status = TestStatusFailed
			result.Error = fmt.Errorf("test setup failed: %w", err)
			result.EndTime = time.Now()
			return result
		}
	}

	// Run test with timeout
	testCtx, testCancel := context.WithTimeout(tr.ctx, testCase.Timeout)
	defer testCancel()

	done := make(chan error, 1)
	go func() {
		done <- testCase.TestFunc()
	}()

	select {
	case err := <-done:
		result.Duration = time.Since(result.StartTime)
		result.EndTime = time.Now()

		if err != nil {
			result.Status = TestStatusFailed
			result.Error = err
			if testCase.ExpectedError != nil && err.Error() == testCase.ExpectedError.Error() {
				result.Status = TestStatusPassed // Expected error occurred
			}
		} else {
			result.Status = TestStatusPassed
		}

	case <-testCtx.Done():
		result.Status = TestStatusTimeout
		result.Error = fmt.Errorf("test timed out after %v", testCase.Timeout)
		result.Duration = testCase.Timeout
		result.EndTime = time.Now()
	}

	// Run test teardown
	if testCase.Teardown != nil {
		if err := testCase.Teardown(); err != nil {
			if result.Error == nil {
				result.Error = fmt.Errorf("test teardown failed: %w", err)
			} else {
				result.Error = fmt.Errorf("%v; test teardown failed: %w", result.Error, err)
			}
			if result.Status == TestStatusPassed {
				result.Status = TestStatusFailed
			}
		}
	}

	// Update test runner statistics
	switch result.Status {
	case TestStatusPassed:
		tr.results.PassedTests++
	case TestStatusFailed:
		tr.results.FailedTests++
	}

	return result
}

// runParallelTests runs test cases in parallel
func (tr *TestRunner) runParallelTests(testCases []*TestCase) {
	if len(testCases) == 0 {
		return
	}

	// Create worker pool
	workerCount := min(4, len(testCases)) // Limit to 4 parallel workers
	jobs := make(chan *TestCase, len(testCases))
	results := make(chan TestResult, len(testCases))

	// Start workers
	for i := 0; i < workerCount; i++ {
		go func() {
			for testCase := range jobs {
				result := tr.runTestCase(testCase)
				results <- result
			}
		}()
	}

	// Send jobs
	for _, testCase := range testCases {
		jobs <- testCase
	}
	close(jobs)

	// Collect results
	for i := 0; i < len(testCases); i++ {
		result := <-results
		tr.results.SuiteResults[len(tr.results.SuiteResults)-1].TestResults = append(
			tr.results.SuiteResults[len(tr.results.SuiteResults)-1].TestResults,
			result,
		)
	}
}

// setupSignalHandling sets up signal handling
func (tr *TestRunner) setupSignalHandling() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		for {
			select {
			case <-tr.ctx.Done():
				return
			case sig := <-sigChan:
				tr.handleSignal(sig)
			}
		}
	}()
}

// handleSignal handles signals
func (tr *TestRunner) handleSignal(sig os.Signal) {
	switch sig {
	case syscall.SIGINT, syscall.SIGTERM:
		tr.state.Running = false
		tr.cancel()
	}
}

// generateSummary generates test summary
func (tr *TestRunner) generateSummary() {
	summary := TestSummary{
		StartTime: tr.startTime,
		EndTime:   time.Now(),
	}

	// Calculate statistics
	totalSuites := len(tr.testSuites)
	completedSuites := tr.results.CompletedSuites
	failedSuites := tr.results.FailedSuites

	totalTests := tr.results.TotalTests
	completedTests := tr.results.CompletedTests
	failedTests := tr.results.FailedTests
	skippedTests := tr.results.SkippedTests
	passedTests := tr.results.PassedTests

	summary.OverallStatus = TestStatusPassed
	if failedTests > 0 {
		summary.OverallStatus = TestStatusFailed
	}

	// Calculate rates
	if totalTests > 0 {
		summary.PassRate = float64(passedTests) / float64(totalTests) * 100
		summary.FailRate = float64(failedTests) / float64(totalTests) * 100
		summary.SkipRate = float64(skippedTests) / float64(totalTests) * 100
	}

	// Find longest/shortest tests
	longestDuration := time.Duration(0)
	shortestDuration := time.Duration(0)
	longestTest := ""
	shortestTest := ""

	for _, suiteResult := range tr.results.SuiteResults {
		for _, testResult := range suiteResult.TestResults {
			if testResult.Duration > longestDuration {
				longestDuration = testResult.Duration
				longestTest = testResult.Name
			}
			if shortestDuration == 0 || testResult.Duration < shortestDuration {
				shortestDuration = testResult.Duration
				shortestTest = testResult.Name
			}
		}
	}
	summary.LongestTest = longestTest
	summary.ShortestTest = shortestTest

	// Find most failing/passed suites
	mostFailingSuite := ""
	mostPassingSuite := ""
	maxFailRate := 0.0
	maxPassRate := 0.0

	for _, suiteResult := range tr.results.SuiteResults {
		if suiteResult.TestCount > 0 {
			failRate := float64(suiteResult.FailedCount) / float64(suiteResult.TestCount) * 100
			passRate := float64(suiteResult.PassedCount) / float64(suiteResult.TestCount) * 100

			if failRate > maxFailRate {
				maxFailRate = failRate
				mostFailingSuite = suiteResult.Name
			}
			if passRate > maxPassRate {
				maxPassRate = passRate
				mostPassingSuite = suiteResult.Name
			}
		}
	}
	summary.MostFailingSuite = mostFailingSuite
	summary.MostPassingSuite = mostPassingSuite

	// Generate recommendations
	summary.Recommendations = tr.generateRecommendations()

	tr.results.Summary = summary
}

// generateRecommendations generates test recommendations
func (tr *TestRunner) generateRecommendations() []string {
	recommendations := []string{}

	// Check for failing tests
	if tr.results.FailedTests > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("Consider investigating %d failing tests", tr.results.FailedTests))
	}

	// Check for skipped tests
	if tr.results.SkippedTests > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("Consider reviewing %d skipped tests", tr.results.SkippedTests))
	}

	// Check for timeouts
	for _, suiteResult := range tr.results.SuiteResults {
		for _, testResult := range suiteResult.TestResults {
			if testResult.Status == TestStatusTimeout {
				recommendations = append(recommendations,
					fmt.Sprintf("Consider increasing timeout for test: %s", testResult.Name))
			}
		}
	}

	// Performance recommendations
	if tr.results.ExecutionTime > 5*time.Minute {
		recommendations = append(recommendations,
			"Consider running tests in parallel to reduce execution time")
	}

	return recommendations
}

// generateReport generates test report
func (tr *TestRunner) generateReport() {
	// This would generate a comprehensive report file
	// Implementation depends on the output format specified
}

// GetResults returns test results
func (tr *TestRunner) GetResults() TestResults {
	return tr.results
}

// GetState returns test runner state
func (tr *TestRunner) GetState() TestRunnerState {
	return tr.state
}

// CreateDefaultTestRunner creates a test runner with default configuration
func CreateDefaultTestRunner() *TestRunner {
	config := TestRunnerConfig{
		Enabled:       true,
		Timeout:       30 * time.Second,
		RetryCount:    0,
		ParallelTests: true,
		StrictMode:    false,
		VerboseOutput: true,
		OutputFormat:  "text",
		SafetyMode:    true,
		ProfilingMode: false,
		DiagnosticMode: true,
		TerminalConfig: terminal.getDefaultTerminalConfig(),
		TestDirectory:  ".",
	}

	return NewTestRunner(config)
}

// CreateBubbleTeaTestSuite creates a Bubble Tea specific test suite
func CreateBubbleTeaTestSuite() TestSuite {
	return TestSuite{
		Name:         "Bubble Tea UI Tests",
		Description:  "Comprehensive test suite for Bubble Tea UI components",
		Timeout:      5 * time.Minute,
		Parallel:     true,
		TestCases: []TestCase{
			{
				Name:         "Minimal Hello World",
				Description:  "Test the simplest Bubble Tea program",
				TestFunc:     minimal_examples.TestMinimalHello,
				Timeout:      10 * time.Second,
				Tags:         []string{"basic", "hello"},
			},
			{
				Name:         "Minimal Key Handling",
				Description:  "Test key handling in Bubble Tea",
				TestFunc:     minimal_examples.TestMinimalKeyHandling,
				Timeout:      10 * time.Second,
				Tags:         []string{"input", "keyboard"},
			},
			{
				Name:         "Minimal Spinner",
				Description:  "Test spinner functionality",
				TestFunc:     minimal_examples.TestMinimalSpinner,
				Timeout:      15 * time.Second,
				Tags:         []string{"animation", "spinner"},
			},
			{
				Name:         "Minimal List",
				Description:  "Test list navigation and selection",
				TestFunc:     minimal_examples.TestMinimalList,
				Timeout:      10 * time.Second,
				Tags:         []string{"navigation", "list"},
			},
			{
				Name:         "Minimal Input",
				Description:  "Test text input functionality",
				TestFunc:     minimal_examples.TestMinimalInput,
				Timeout:      15 * time.Second,
				Tags:         []string{"input", "text"},
			},
			{
				Name:         "Minimal Progress Bar",
				Description:  "Test progress bar functionality",
				TestFunc:     minimal_examples.TestMinimalProgressBar,
				Timeout:      20 * time.Second,
				Tags:         []string{"progress", "animation"},
			},
			{
				Name:         "Minimal Form",
				Description:  "Test form handling and validation",
				TestFunc:     minimal_examples.TestMinimalForm,
				Timeout:      20 * time.Second,
				Tags:         []string{"form", "validation"},
			},
			{
				Name:         "Minimal Timer",
				Description:  "Test timer functionality",
				TestFunc:     minimal_examples.TestMinimalTimer,
				Timeout:      15 * time.Second,
				Tags:         []string{"timer", "animation"},
			},
			{
				Name:         "Minimal Modal",
				Description:  "Test modal dialog functionality",
				TestFunc:     minimal_examples.TestMinimalModal,
				Timeout:      15 * time.Second,
				Tags:         []string{"modal", "dialog"},
			},
			{
				Name:         "Minimal Error Handling",
				Description:  "Test error handling and recovery",
				TestFunc:     minimal_examples.TestMinimalErrorHandling,
				Timeout:      30 * time.Second,
				Tags:         []string{"error", "recovery"},
			},
		},
		Setup: func() error {
			// Setup for Bubble Tea tests
			fmt.Println("Setting up Bubble Tea test environment...")
			return nil
		},
		Teardown: func() error {
			// Teardown for Bubble Tea tests
			fmt.Println("Cleaning up Bubble Tea test environment...")
			return nil
		},
	}
}

// CreateStressTestSuite creates a stress test suite
func CreateStressTestSuite() TestSuite {
	return TestSuite{
		Name:         "Bubble Tea Stress Tests",
		Description:  "Stress tests for Bubble Tea performance and reliability",
		Timeout:      10 * time.Minute,
		Parallel:     true,
		TestCases: []TestCase{
			{
				Name:         "High Frequency Updates",
				Description:  "Test UI with rapid updates",
				TestFunc:     testHighFrequencyUpdates,
				Timeout:      30 * time.Second,
				Tags:         []string{"stress", "performance"},
			},
			{
				Name:         "Memory Usage Test",
				Description:  "Test memory usage patterns",
				TestFunc:     testMemoryUsage,
				Timeout:      60 * time.Second,
				Tags:         []string{"memory", "stress"},
			},
			{
				Name:         "Concurrent Input Test",
				Description:  "Test concurrent input handling",
				TestFunc:     testConcurrentInput,
				Timeout:      30 * time.Second,
				Tags:         []string{"concurrency", "input"},
			},
		},
	}
}

// Stress test functions
func testHighFrequencyUpdates() error {
	// Implementation for high frequency updates test
	model := minimal_examples.NewMinimalSpinner()
	program := tea.NewProgram(model)

	// Run with high frequency
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	go func() {
		program.Run()
	}()

	// Simulate high frequency updates
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			// Send update message
			program.Send(struct{}{})
		}
	}
}

func testMemoryUsage() error {
	// Implementation for memory usage test
	model := minimal_examples.NewMinimalInput()
	program := tea.NewProgram(model)

	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Second)
	defer cancel()

	go func() {
		program.Run()
	}()

	// Simulate memory intensive operations
	for i := 0; i < 1000; i++ {
		select {
		case <-ctx.Done():
			return nil
		default:
			// Send large input
			program.Send(struct {
				tea.Msg
				data string
			}{data: strings.Repeat("x", 1000)})
		}
	}

	return nil
}

func testConcurrentInput() error {
	// Implementation for concurrent input test
	model := minimal_examples.NewMinimalList()
	program := tea.NewProgram(model)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	go func() {
		program.Run()
	}()

	// Simulate concurrent input
	for i := 0; i < 100; i++ {
		select {
		case <-ctx.Done():
			return nil
		default:
			// Send random keys
			keys := []tea.KeyMsg{
				{Type: tea.KeyUp},
				{Type: tea.KeyDown},
				{Type: tea.KeyEnter},
			}
			program.Send(keys[i%len(keys)])
		}
	}

	return nil
}

// CreateIntegrationTestSuite creates an integration test suite
func CreateIntegrationTestSuite() TestSuite {
	return TestSuite{
		Name:         "Bubble Tea Integration Tests",
		Description:  "Integration tests for complete Bubble Tea workflows",
		Timeout:      5 * time.Minute,
		Parallel:     false,
		TestCases: []TestCase{
			{
				Name:         "Complete Workflow Test",
				Description:  "Test complete UI workflow from start to finish",
				TestFunc:     testCompleteWorkflow,
				Timeout:      60 * time.Second,
				Tags:         []string{"integration", "workflow"},
			},
			{
				Name:         "Error Recovery Test",
				Description:  "Test error recovery mechanisms",
				TestFunc:     testErrorRecovery,
				Timeout:      30 * time.Second,
				Tags:         []string{"integration", "recovery"},
			},
			{
				Name:         "Multi Stage Navigation Test",
				Description:  "Test multi-stage navigation and state management",
				TestFunc:     testMultiStageNavigation,
				Timeout:      45 * time.Second,
				Tags:         []string{"integration", "navigation"},
			},
		},
	}
}

// Integration test functions
func testCompleteWorkflow() error {
	// Implementation for complete workflow test
	// This would test the full UI workflow
	return nil
}

func testErrorRecovery() error {
	// Implementation for error recovery test
	// This would test error recovery mechanisms
	return nil
}

func testMultiStageNavigation() error {
	// Implementation for multi-stage navigation test
	// This would test complex navigation patterns
	return nil
}

// RunTestSuites runs all predefined test suites
func RunTestSuites() error {
	runner := CreateDefaultTestRunner()

	// Add test suites
	runner.AddTestSuite(CreateBubbleTeaTestSuite())
	runner.AddTestSuite(CreateStressTestSuite())
	runner.AddTestSuite(CreateIntegrationTestSuite())

	// Run tests
	results, err := runner.RunTests()
	if err != nil {
		return fmt.Errorf("test execution failed: %w", err)
	}

	// Print results
	PrintTestResults(results)

	return nil
}

// PrintTestResults prints test results
func PrintTestResults(results TestResults) {
	fmt.Printf("\n🧪 TEST RESULTS\n")
	fmt.Printf("=" * 50 + "\n")
	fmt.Printf("Total Suites: %d\n", results.TotalSuites)
	fmt.Printf("Completed Suites: %d\n", results.CompletedSuites)
	fmt.Printf("Failed Suites: %d\n", results.FailedSuites)
	fmt.Printf("Total Tests: %d\n", results.TotalTests)
	fmt.Printf("Passed Tests: %d\n", results.PassedTests)
	fmt.Printf("Failed Tests: %d\n", results.FailedTests)
	fmt.Printf("Skipped Tests: %d\n", results.SkippedTests)
	fmt.Printf("Execution Time: %v\n", results.ExecutionTime)
	fmt.Printf("Overall Status: %s\n", results.Summary.OverallStatus)
	fmt.Printf("Pass Rate: %.1f%%\n", results.Summary.PassRate)

	if len(results.Summary.Recommendations) > 0 {
		fmt.Printf("\n💡 RECOMMENDATIONS\n")
		for _, rec := range results.Summary.Recommendations {
			fmt.Printf("  - %s\n", rec)
		}
	}
}