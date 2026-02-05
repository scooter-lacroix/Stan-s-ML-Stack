// internal/utils/logger.go
package utils

import (
	"os"
	"path/filepath"
	"time"
)

// Logger provides simple logging functionality
type Logger struct {
	logDir string
}

// NewLogger creates a new logger instance
func NewLogger() (*Logger, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	logDir := filepath.Join(homeDir, ".mlstack", "logs")
	err = os.MkdirAll(logDir, 0755)
	if err != nil {
		return nil, err
	}

	return &Logger{logDir: logDir}, nil
}

// GetLogDir returns the log directory path
func (l *Logger) GetLogDir() string {
	return l.logDir
}

// CreateLogFile creates a new log file with timestamp
func (l *Logger) CreateLogFile(componentID string) (*os.File, error) {
	timestamp := time.Now().Format("20060102_150405")
	logPath := filepath.Join(l.logDir, componentID+"_"+timestamp+".log")

	return os.Create(logPath)
}
