// internal/ui/monitoring/animations.go
package monitoring

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
)

// AnimationType represents different types of animations
type AnimationType int

const (
	AnimationTypeFade AnimationType = iota
	AnimationTypeSlide
	AnimationTypeScale
	AnimationTypeRotate
	AnimationTypeBounce
	AnimationTypePulse
	AnimationTypeTypewriter
	AnimationTypeProgress
	AnimationTypeParticle
	AnimationTypeWave
)

// AnimationState represents the current state of an animation
type AnimationState int

const (
	AnimationStateIdle AnimationState = iota
	AnimationStateRunning
	AnimationStatePaused
	AnimationStateCompleted
	AnimationStateCancelled
)

// AnimationConfig holds configuration for animations
type AnimationConfig struct {
	Type        AnimationType
	Duration    time.Duration
	Delay       time.Duration
	Easing      EasingFunction
	Loop        bool
	AutoReverse bool
	Direction   AnimationDirection
	From        AnimationValues
	To          AnimationValues
}

// AnimationDirection represents the direction of an animation
type AnimationDirection int

const (
	DirectionForward AnimationDirection = iota
	DirectionReverse
	DirectionAlternate
)

// AnimationValues holds values that can be animated
type AnimationValues struct {
	Opacity    float64
	ScaleX     float64
	ScaleY     float64
	Rotate     float64
	TranslateX float64
	TranslateY float64
	Color      lipgloss.Color
	Width      int
	Height     int
}

// Animation represents a single animation
type Animation struct {
	Config      AnimationConfig
	State       AnimationState
	StartTime   time.Time
	PauseTime   time.Time
	CurrentTime float64
	Values      AnimationValues
	OnComplete  func()
	OnUpdate    func(progress float64, values AnimationValues)
	Theme       *AMDTheme
}

// AnimationManager manages multiple animations
type AnimationManager struct {
	animations  []*Animation
	running     bool
	lastUpdate  time.Time
	theme       *AMDTheme
	maxAnimations int
}

// TransitionEffect represents visual transition effects
type TransitionEffect struct {
	Type        TransitionType
	Duration    time.Duration
	Easing      EasingFunction
	From        TransitionValues
	To          TransitionValues
	Theme       *AMDTheme
}

// TransitionType represents different transition types
type TransitionType int

const (
	TransitionTypeFade TransitionType = iota
	TransitionTypeSlideLeft
	TransitionTypeSlideRight
	TransitionTypeSlideUp
	TransitionTypeSlideDown
	TransitionTypeScale
	TransitionTypeFlip
	TransitionTypeDissolve
	TransitionTypeWipe
	TransitionTypePush
)

// TransitionValues holds values for transitions
type TransitionValues struct {
	Content     string
	Style       lipgloss.Style
	Opacity     float64
	Position    Position
	Clip        ClipRect
	Blur        float64
	Brightness  float64
	Contrast    float64
}

// Position represents a position in 2D space
type Position struct {
	X float64
	Y float64
}

// ClipRect represents a clipping rectangle
type ClipRect struct {
	X, Y, Width, Height float64
}

// ParticleEffect represents particle system effects
type ParticleEffect struct {
	Particles   []*Particle
	Config      ParticleConfig
	Theme       *AMDTheme
	IsRunning   bool
	StartTime   time.Time
}

// ParticleConfig holds configuration for particle effects
type ParticleConfig struct {
	Count       int
	Lifetime    time.Duration
	Speed       float64
	Direction   float64
	Spread      float64
	Gravity     float64
	Size        float64
	Color       lipgloss.Color
	Shape       ParticleShape
	Emission    EmissionType
}

// ParticleShape represents different particle shapes
type ParticleShape int

const (
	ShapeCircle ParticleShape = iota
	ShapeSquare
	ShapeTriangle
	ShapeStar
	ShapeDot
)

// EmissionType represents different particle emission types
type EmissionType int

const (
	EmissionPoint EmissionType = iota
	EmissionLine
	EmissionCircle
	EmissionRectangle
	EmissionBurst
)

// Particle represents a single particle
type Particle struct {
	Position    Position
	Velocity    Position
	Acceleration Position
	Life        float64
	MaxLife     float64
	Size        float64
	Color       lipgloss.Color
	Opacity     float64
	Shape       ParticleShape
	Active      bool
}

// LoadingAnimation represents loading spinner animations
type LoadingAnimation struct {
	Type        LoadingType
	Frame       int
	FrameCount  int
	LastUpdate  time.Time
	Speed       time.Duration
	Theme       *AMDTheme
	Text        string
	IsRunning   bool
}

// LoadingType represents different loading animation types
type LoadingType int

const (
	LoadingTypeSpinner LoadingType = iota
	LoadingTypeDots
	LoadingTypeBars
	LoadingTypePulse
	LoadingTypeWave
	LoadingTypeOrbit
	LoadingTypeMeter
	LoadingTypeClock
)

// NewAnimation creates a new animation
func NewAnimation(config AnimationConfig, theme *AMDTheme) *Animation {
	return &Animation{
		Config:    config,
		State:     AnimationStateIdle,
		Values:    config.From,
		Theme:     theme,
	}
}

// Start starts the animation
func (a *Animation) Start() {
	a.State = AnimationStateRunning
	a.StartTime = time.Now()
	a.CurrentTime = 0
}

// Pause pauses the animation
func (a *Animation) Pause() {
	if a.State == AnimationStateRunning {
		a.State = AnimationStatePaused
		a.PauseTime = time.Now()
	}
}

// Resume resumes a paused animation
func (a *Animation) Resume() {
	if a.State == AnimationStatePaused {
		a.State = AnimationStateRunning
		pauseDuration := time.Since(a.PauseTime)
		a.StartTime = a.StartTime.Add(pauseDuration)
	}
}

// Stop stops the animation
func (a *Animation) Stop() {
	a.State = AnimationStateCompleted
	if a.OnComplete != nil {
		a.OnComplete()
	}
}

// Cancel cancels the animation
func (a *Animation) Cancel() {
	a.State = AnimationStateCancelled
}

// Update updates the animation state
func (a *Animation) Update(deltaTime time.Duration) {
	if a.State != AnimationStateRunning {
		return
	}

	// Apply delay
	if time.Since(a.StartTime) < a.Config.Delay {
		return
	}

	// Calculate progress
	elapsed := time.Since(a.StartTime) - a.Config.Delay
	progress := float64(elapsed) / float64(a.Config.Duration)

	if progress >= 1.0 {
		progress = 1.0

		if a.Config.Loop {
			// Restart animation
			a.StartTime = time.Now()
			if a.Config.AutoReverse {
				// Swap from and to values
				a.Config.From, a.Config.To = a.Config.To, a.Config.From
			}
		} else {
			// Complete animation
			a.State = AnimationStateCompleted
			if a.OnComplete != nil {
				a.OnComplete()
			}
		}
	}

	// Apply easing function
	easedProgress := a.Config.Easing(progress)

	// Update values based on animation type
	a.updateValues(easedProgress)
	a.CurrentTime = easedProgress

	// Call update callback
	if a.OnUpdate != nil {
		a.OnUpdate(easedProgress, a.Values)
	}
}

// updateValues updates animation values based on progress
func (a *Animation) updateValues(progress float64) {
	from := a.Config.From
	to := a.Config.To

	switch a.Config.Type {
	case AnimationTypeFade:
		a.Values.Opacity = from.Opacity + (to.Opacity-from.Opacity)*progress

	case AnimationTypeSlide:
		a.Values.TranslateX = from.TranslateX + (to.TranslateX-from.TranslateX)*progress
		a.Values.TranslateY = from.TranslateY + (to.TranslateY-from.TranslateY)*progress

	case AnimationTypeScale:
		a.Values.ScaleX = from.ScaleX + (to.ScaleX-from.ScaleX)*progress
		a.Values.ScaleY = from.ScaleY + (to.ScaleY-from.ScaleY)*progress

	case AnimationTypeRotate:
		a.Values.Rotate = from.Rotate + (to.Rotate-from.Rotate)*progress

	case AnimationTypeBounce:
		// Apply bounce effect
		bounceProgress := a.bounceEasing(progress)
		a.Values.TranslateY = from.TranslateY + (to.TranslateY-from.TranslateY)*bounceProgress

	case AnimationTypePulse:
		// Apply pulse effect
		pulseProgress := math.Sin(progress * math.Pi * 2)
		scale := 1.0 + pulseProgress*0.1
		a.Values.ScaleX = scale
		a.Values.ScaleY = scale

	case AnimationTypeProgress:
		// Update width based on progress
		a.Values.Width = int(float64(from.Width) + (float64(to.Width)-float64(from.Width))*progress)

	default:
		// Default to linear interpolation
		a.Values.Opacity = from.Opacity + (to.Opacity-from.Opacity)*progress
		a.Values.ScaleX = from.ScaleX + (to.ScaleX-from.ScaleX)*progress
		a.Values.ScaleY = from.ScaleY + (to.ScaleY-from.ScaleY)*progress
	}
}

// bounceEasing applies bounce easing function
func (a *Animation) bounceEasing(t float64) float64 {
	if t < 0.5 {
		return 8 * t * t
	}
	return 1 - 8*math.Pow(t-1, 2)
}

// GetCurrentValue returns the current value for a specific property
func (a *Animation) GetCurrentValue(property string) float64 {
	switch property {
	case "opacity":
		return a.Values.Opacity
	case "scaleX":
		return a.Values.ScaleX
	case "scaleY":
		return a.Values.ScaleY
	case "rotate":
		return a.Values.Rotate
	case "translateX":
		return a.Values.TranslateX
	case "translateY":
		return a.Values.TranslateY
	default:
		return 0
	}
}

// IsCompleted returns true if the animation is completed
func (a *Animation) IsCompleted() bool {
	return a.State == AnimationStateCompleted
}

// IsRunning returns true if the animation is currently running
func (a *Animation) IsRunning() bool {
	return a.State == AnimationStateRunning
}

// NewAnimationManager creates a new animation manager
func NewAnimationManager(theme *AMDTheme, maxAnimations int) *AnimationManager {
	return &AnimationManager{
		animations:    make([]*Animation, 0, maxAnimations),
		running:       false,
		lastUpdate:    time.Now(),
		theme:         theme,
		maxAnimations: maxAnimations,
	}
}

// Add adds an animation to the manager
func (m *AnimationManager) Add(animation *Animation) bool {
	if len(m.animations) >= m.maxAnimations {
		return false
	}

	m.animations = append(m.animations, animation)
	if !m.running {
		m.running = true
	}

	return true
}

// Remove removes an animation from the manager
func (m *AnimationManager) Remove(animation *Animation) {
	for i, anim := range m.animations {
		if anim == animation {
			m.animations = append(m.animations[:i], m.animations[i+1:]...)
			break
		}
	}

	if len(m.animations) == 0 {
		m.running = false
	}
}

// Update updates all managed animations
func (m *AnimationManager) Update() {
	now := time.Now()
	deltaTime := now.Sub(m.lastUpdate)
	m.lastUpdate = now

	for i := len(m.animations) - 1; i >= 0; i-- {
		animation := m.animations[i]
		animation.Update(deltaTime)

		// Remove completed animations
		if animation.IsCompleted() && !animation.Config.Loop {
			m.animations = append(m.animations[:i], m.animations[i+1:]...)
		}
	}

	if len(m.animations) == 0 {
		m.running = false
	}
}

// GetActiveCount returns the number of active animations
func (m *AnimationManager) GetActiveCount() int {
	count := 0
	for _, animation := range m.animations {
		if animation.IsRunning() {
			count++
		}
	}
	return count
}

// Clear clears all animations
func (m *AnimationManager) Clear() {
	m.animations = m.animations[:0]
	m.running = false
}

// NewTransitionEffect creates a new transition effect
func NewTransitionEffect(effectType TransitionType, duration time.Duration, theme *AMDTheme) *TransitionEffect {
	return &TransitionEffect{
		Type:     effectType,
		Duration: duration,
		Easing:   EaseOutCubic,
		From:     TransitionValues{Opacity: 1.0},
		To:       TransitionValues{Opacity: 1.0},
		Theme:    theme,
	}
}

// Apply applies the transition effect to content
func (t *TransitionEffect) Apply(content string, progress float64) string {
	easedProgress := t.Easing(progress)

	switch t.Type {
	case TransitionTypeFade:
		return t.applyFade(content, easedProgress)
	case TransitionTypeSlideLeft:
		return t.applySlide(content, easedProgress, -1, 0)
	case TransitionTypeSlideRight:
		return t.applySlide(content, easedProgress, 1, 0)
	case TransitionTypeSlideUp:
		return t.applySlide(content, easedProgress, 0, -1)
	case TransitionTypeSlideDown:
		return t.applySlide(content, easedProgress, 0, 1)
	case TransitionTypeScale:
		return t.applyScale(content, easedProgress)
	case TransitionTypeDissolve:
		return t.applyDissolve(content, easedProgress)
	case TransitionTypeWipe:
		return t.applyWipe(content, easedProgress)
	default:
		return content
	}
}

// applyFade applies fade transition
func (t *TransitionEffect) applyFade(content string, progress float64) string {
	opacity := t.From.Opacity + (t.To.Opacity-t.From.Opacity)*progress
	if opacity <= 0 {
		return ""
	}

	// Apply opacity by modifying style
	style := lipgloss.NewStyle().Foreground(
		lipgloss.Color(fmt.Sprintf("#%02x%02x%02x",
			int(255*opacity),
			int(255*opacity),
			int(255*opacity))),
	)

	return style.Render(content)
}

// applySlide applies slide transition
func (t *TransitionEffect) applySlide(content string, progress float64, dirX, dirY float64) string {
	lines := strings.Split(content, "\n")
	width := 0
	height := len(lines)

	for _, line := range lines {
		if len(line) > width {
			width = len(line)
		}
	}

	// Calculate offset
	offsetX := int(float64(width) * progress * dirX)
	offsetY := int(float64(height) * progress * dirY)

	// Create slide effect
	var result strings.Builder
	for i, line := range lines {
		if offsetY <= i && i < offsetY+height {
			if offsetX >= 0 && offsetX < len(line) {
				visibleLine := line[minInt(maxInt(0, offsetX), len(line)):]
				result.WriteString(visibleLine)
			} else if offsetX < 0 {
				visibleLength := minInt(len(line), -offsetX)
				visibleLine := line[:visibleLength]
				prefix := strings.Repeat(" ", -offsetX)
				result.WriteString(prefix + visibleLine)
			}
		}
		if i < len(lines)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}

// applyScale applies scale transition
func (t *TransitionEffect) applyScale(content string, progress float64) string {
	// Simplified scale effect
	scale := 0.5 + progress*0.5

	lines := strings.Split(content, "\n")
	centerX := len(lines[0]) / 2
	centerY := len(lines) / 2

	var result strings.Builder
	for y, line := range lines {
		scaledY := int(float64(y-centerY)*scale + float64(centerY))
		if scaledY >= 0 && scaledY < len(lines) {
			scaledLine := t.scaleLine(line, scale, centerX)
			result.WriteString(scaledLine)
		}
		if y < len(lines)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}

// scaleLine scales a single line
func (t *TransitionEffect) scaleLine(line string, scale float64, centerX int) string {
	if scale >= 1.0 {
		return line
	}

	start := int(float64(centerX) - float64(centerX)*scale)
	end := int(float64(centerX) + float64(len(line)-centerX)*scale)

	if start < 0 {
		start = 0
	}
	if end > len(line) {
		end = len(line)
	}

	if start >= end {
		return ""
	}

	return line[start:end]
}

// applyDissolve applies dissolve transition
func (t *TransitionEffect) applyDissolve(content string, progress float64) string {
	// Create dissolve effect by randomly removing characters
	lines := strings.Split(content, "\n")
	var result strings.Builder

	seed := int(progress * 1000) // Simple seed based on progress

	for y, line := range lines {
		for x, char := range line {
			// Simple pseudo-random based on position and seed
			shouldShow := (x*y+seed)%100 > int(progress*100)
			if shouldShow {
				result.WriteRune(char)
			} else {
				result.WriteRune(' ')
			}
		}
		if y < len(lines)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}

// applyWipe applies wipe transition
func (t *TransitionEffect) applyWipe(content string, progress float64) string {
	lines := strings.Split(content, "\n")
	width := 0
	for _, line := range lines {
		if len(line) > width {
			width = len(line)
		}
	}

	wipePosition := int(float64(width) * progress)

	var result strings.Builder
	for _, line := range lines {
		if wipePosition <= len(line) {
			result.WriteString(line[:wipePosition])
		} else {
			result.WriteString(line)
		}
		result.WriteString("\n")
	}

	return result.String()
}

// NewParticleEffect creates a new particle effect
func NewParticleEffect(config ParticleConfig, theme *AMDTheme) *ParticleEffect {
	effect := &ParticleEffect{
		Config:    config,
		Theme:     theme,
		Particles: make([]*Particle, config.Count),
	}

	// Initialize particles
	for i := 0; i < config.Count; i++ {
		effect.Particles[i] = &Particle{
			Position:    Position{X: 0, Y: 0},
			Velocity:    Position{X: 0, Y: 0},
			Acceleration: Position{X: 0, Y: config.Gravity},
			Life:        1.0,
			MaxLife:     1.0,
			Size:        config.Size,
			Color:       config.Color,
			Opacity:     1.0,
			Shape:       config.Shape,
			Active:      false,
		}
	}

	return effect
}

// Start starts the particle effect
func (p *ParticleEffect) Start() {
	p.IsRunning = true
	p.StartTime = time.Now()

	// Initialize particle positions based on emission type
	for i, particle := range p.Particles {
		p.initializeParticle(particle, i)
		particle.Active = true
	}
}

// Stop stops the particle effect
func (p *ParticleEffect) Stop() {
	p.IsRunning = false
}

// Update updates the particle effect
func (p *ParticleEffect) Update(deltaTime time.Duration) {
	if !p.IsRunning {
		return
	}

	for _, particle := range p.Particles {
		if !particle.Active {
			continue
		}

		// Update physics
		particle.Velocity.X += particle.Acceleration.X * deltaTime.Seconds()
		particle.Velocity.Y += particle.Acceleration.Y * deltaTime.Seconds()
		particle.Position.X += particle.Velocity.X * deltaTime.Seconds()
		particle.Position.Y += particle.Velocity.Y * deltaTime.Seconds()

		// Update life
		particle.Life -= deltaTime.Seconds() / p.Config.Lifetime.Seconds()
		particle.Opacity = particle.Life

		// Deactivate dead particles
		if particle.Life <= 0 {
			particle.Active = false
		}
	}

	// Check if all particles are dead
	allDead := true
	for _, particle := range p.Particles {
		if particle.Active {
			allDead = false
			break
		}
	}

	if allDead {
		p.Stop()
	}
}

// initializeParticle initializes a particle based on emission type
func (p *ParticleEffect) initializeParticle(particle *Particle, index int) {
	switch p.Config.Emission {
	case EmissionPoint:
		particle.Position = Position{X: 0, Y: 0}

	case EmissionBurst:
		angle := float64(index) / float64(p.Config.Count) * 2 * math.Pi
		speed := p.Config.Speed
		particle.Position = Position{X: 0, Y: 0}
		particle.Velocity = Position{
			X: math.Cos(angle) * speed,
			Y: math.Sin(angle) * speed,
		}

	case EmissionCircle:
		angle := float64(index) / float64(p.Config.Count) * 2 * math.Pi
		radius := 5.0
		particle.Position = Position{
			X: math.Cos(angle) * radius,
			Y: math.Sin(angle) * radius,
		}

	default:
		particle.Position = Position{X: 0, Y: 0}
	}

	// Add randomness
	if p.Config.Spread > 0 {
		spread := p.Config.Spread / 2
		particle.Velocity.X += (float64(index%100) / 100.0 - 0.5) * spread
		particle.Velocity.Y += (float64(index%100) / 100.0 - 0.5) * spread
	}
}

// Render renders the particle effect
func (p *ParticleEffect) Render() string {
	if !p.IsRunning {
		return ""
	}

	// Create a simple ASCII representation
	width, height := 40, 20
	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Render particles
	for _, particle := range p.Particles {
		if !particle.Active {
			continue
		}

		x := int(particle.Position.X + float64(width)/2)
		y := int(particle.Position.Y + float64(height)/2)

		if x >= 0 && x < width && y >= 0 && y < height {
			char := p.getParticleChar(particle)
			grid[y][x] = char
		}
	}

	// Convert grid to string
	var result strings.Builder
	for _, row := range grid {
		result.WriteString(string(row))
		result.WriteString("\n")
	}

	return result.String()
}

// getParticleChar returns the character for a particle
func (p *ParticleEffect) getParticleChar(particle *Particle) rune {
	switch particle.Shape {
	case ShapeCircle:
		return '●'
	case ShapeSquare:
		return '■'
	case ShapeTriangle:
		return '▲'
	case ShapeStar:
		return '★'
	case ShapeDot:
		return '·'
	default:
		return '•'
	}
}

// NewLoadingAnimation creates a new loading animation
func NewLoadingAnimation(animType LoadingType, text string, theme *AMDTheme) *LoadingAnimation {
	var frameCount int
	var speed time.Duration

	switch animType {
	case LoadingTypeSpinner:
		frameCount = 8
		speed = 100 * time.Millisecond
	case LoadingTypeDots:
		frameCount = 3
		speed = 500 * time.Millisecond
	case LoadingTypeBars:
		frameCount = 10
		speed = 200 * time.Millisecond
	case LoadingTypePulse:
		frameCount = 20
		speed = 100 * time.Millisecond
	case LoadingTypeWave:
		frameCount = 8
		speed = 150 * time.Millisecond
	case LoadingTypeOrbit:
		frameCount = 8
		speed = 100 * time.Millisecond
	case LoadingTypeMeter:
		frameCount = 100
		speed = 50 * time.Millisecond
	case LoadingTypeClock:
		frameCount = 12
		speed = 100 * time.Millisecond
	default:
		frameCount = 4
		speed = 250 * time.Millisecond
	}

	return &LoadingAnimation{
		Type:       animType,
		Frame:      0,
		FrameCount: frameCount,
		Speed:      speed,
		Theme:      theme,
		Text:       text,
		IsRunning:  false,
	}
}

// Start starts the loading animation
func (l *LoadingAnimation) Start() {
	l.IsRunning = true
	l.LastUpdate = time.Now()
	l.Frame = 0
}

// Stop stops the loading animation
func (l *LoadingAnimation) Stop() {
	l.IsRunning = false
}

// Update updates the loading animation
func (l *LoadingAnimation) Update() {
	if !l.IsRunning {
		return
	}

	now := time.Now()
	if now.Sub(l.LastUpdate) >= l.Speed {
		l.Frame = (l.Frame + 1) % l.FrameCount
		l.LastUpdate = now
	}
}

// Render renders the loading animation
func (l *LoadingAnimation) Render() string {
	if !l.IsRunning {
		return l.Text
	}

	switch l.Type {
	case LoadingTypeSpinner:
		return l.renderSpinner()
	case LoadingTypeDots:
		return l.renderDots()
	case LoadingTypeBars:
		return l.renderBars()
	case LoadingTypePulse:
		return l.renderPulse()
	case LoadingTypeWave:
		return l.renderWave()
	case LoadingTypeOrbit:
		return l.renderOrbit()
	case LoadingTypeMeter:
		return l.renderMeter()
	case LoadingTypeClock:
		return l.renderClock()
	default:
		return l.Text
	}
}

// renderSpinner renders spinner animation
func (l *LoadingAnimation) renderSpinner() string {
	spinners := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	spinner := spinners[l.Frame%len(spinners)]
	return fmt.Sprintf("%s %s", spinner, l.Text)
}

// renderDots renders dots animation
func (l *LoadingAnimation) renderDots() string {
	dots := ""
	for i := 0; i < 3; i++ {
		if i <= l.Frame%3 {
			dots += "●"
		} else {
			dots += "○"
		}
	}
	return fmt.Sprintf("%s %s", dots, l.Text)
}

// renderBars renders bars animation
func (l *LoadingAnimation) renderBars() string {
	barCount := 10
	filled := l.Frame % (barCount + 1)

	bars := ""
	for i := 0; i < barCount; i++ {
		if i < filled {
			bars += "█"
		} else {
			bars += "░"
		}
	}

	// Add percentage
	percentage := int(float64(filled) / float64(barCount) * 100)
	return fmt.Sprintf("[%s] %d%% %s", bars, percentage, l.Text)
}

// renderPulse renders pulse animation
func (l *LoadingAnimation) renderPulse() string {
	progress := float64(l.Frame) / float64(l.FrameCount)
	pulse := math.Sin(progress * math.Pi * 2) * 0.5 + 0.5

	// Create pulsing text
	style := lipgloss.NewStyle().
		Bold(pulse > 0.5).
		Foreground(lipgloss.Color(fmt.Sprintf("#%02x%02x%02x",
			int(255*pulse),
			int(100+155*pulse),
			int(100))))

	return style.Render(l.Text)
}

// renderWave renders wave animation
func (l *LoadingAnimation) renderWave() string {
	text := l.Text
	waveText := ""

	for i, char := range text {
		offset := math.Sin(float64(i+l.Frame)/2.0) * 0.5 + 0.5
		if offset > 0.5 {
			waveText += string(char)
		} else {
			waveText += " "
		}
	}

	return waveText
}

// renderOrbit renders orbit animation
func (l *LoadingAnimation) renderOrbit() string {
	angles := []float64{0, 45, 90, 135, 180, 225, 270, 315}
	currentAngle := float64(l.Frame) / float64(l.FrameCount) * 360

	orbit := "["
	for _, angle := range angles {
		diff := math.Abs(angle-currentAngle)
		if diff < 45 {
			orbit += "●"
		} else {
			orbit += "○"
		}
	}
	orbit += "]"

	return fmt.Sprintf("%s %s", orbit, l.Text)
}

// renderMeter renders meter animation
func (l *LoadingAnimation) renderMeter() string {
	progress := float64(l.Frame) / float64(l.FrameCount)
	meterWidth := 20
	filled := int(progress * float64(meterWidth))

	meter := ""
	for i := 0; i < meterWidth; i++ {
		if i < filled {
			meter += "█"
		} else {
			meter += "░"
		}
	}

	percentage := int(progress * 100)
	return fmt.Sprintf("[%s] %d%% %s", meter, percentage, l.Text)
}

// renderClock renders clock animation
func (l *LoadingAnimation) renderClock() string {
	hour := (l.Frame / 5) % 12
	minute := l.Frame % 12 * 5

	clock := fmt.Sprintf("%02d:%02d", hour, minute)
	return fmt.Sprintf("🕐 %s %s", clock, l.Text)
}

// CreateNotificationAnimation creates a notification animation
func CreateNotificationAnimation(message string, duration time.Duration, theme *AMDTheme) *Animation {
	config := AnimationConfig{
		Type:     AnimationTypeSlide,
		Duration: duration,
		Easing:   EaseOutBounce,
		From: AnimationValues{
			TranslateY: -10,
			Opacity:    0,
		},
		To: AnimationValues{
			TranslateY: 0,
			Opacity:    1,
		},
	}

	animation := NewAnimation(config, theme)

	// Set up auto-reverse for dismiss animation
	animation.OnComplete = func() {
		// Create reverse animation for dismiss
		reverseConfig := config
		reverseConfig.From, reverseConfig.To = config.To, config.From
		reverseAnimation := NewAnimation(reverseConfig, theme)
		reverseAnimation.Start()
	}

	return animation
}

// CreateProgressAnimation creates a progress animation
func CreateProgressAnimation(fromValue, toValue float64, duration time.Duration, theme *AMDTheme) *Animation {
	config := AnimationConfig{
		Type:     AnimationTypeProgress,
		Duration: duration,
		Easing:   EaseOutCubic,
		From: AnimationValues{
			Width: int(fromValue),
		},
		To: AnimationValues{
			Width: int(toValue),
		},
	}

	return NewAnimation(config, theme)
}

// CreateHeartbeatAnimation creates a heartbeat animation for indicators
func CreateHeartbeatAnimation(theme *AMDTheme) *Animation {
	config := AnimationConfig{
		Type:        AnimationTypePulse,
		Duration:    1000 * time.Millisecond,
		Loop:        true,
		Easing:      EaseInOutSine,
		AutoReverse: true,
		From: AnimationValues{
			ScaleX: 1.0,
			ScaleY: 1.0,
			Opacity: 0.7,
		},
		To: AnimationValues{
			ScaleX: 1.2,
			ScaleY: 1.2,
			Opacity: 1.0,
		},
	}

	return NewAnimation(config, theme)
}

// CreateTypewriterAnimation creates a typewriter effect for text
func CreateTypewriterAnimation(text string, charDuration time.Duration, theme *AMDTheme) *Animation {
	totalDuration := time.Duration(len(text)) * charDuration

	config := AnimationConfig{
		Type:     AnimationTypeTypewriter,
		Duration: totalDuration,
		Easing:   EaseLinear,
		From: AnimationValues{
			Width: 0,
		},
		To: AnimationValues{
			Width: len(text),
		},
	}

	animation := NewAnimation(config, theme)

	// Custom update callback for typewriter effect
	animation.OnUpdate = func(progress float64, values AnimationValues) {
		// This would be handled by the rendering system
	}

	return animation
}

// Additional easing functions
var (
	EaseInSine EasingFunction = func(t float64) float64 {
		return 1 - math.Cos((t*math.Pi)/2)
	}
	EaseOutSine EasingFunction = func(t float64) float64 {
		return math.Sin((t * math.Pi) / 2)
	}
	EaseInOutSine EasingFunction = func(t float64) float64 {
		return -(math.Cos(math.Pi*t) - 1) / 2
	}
	EaseOutBounce EasingFunction = func(t float64) float64 {
		n1 := 7.5625
		d1 := 2.75

		if t < 1/d1 {
			return n1 * t * t
		} else if t < 2/d1 {
			t -= 1.5 / d1
			return n1*t*t + 0.75
		} else if t < 2.5/d1 {
			t -= 2.25 / d1
			return n1*t*t + 0.9375
		} else {
			t -= 2.65 / d1
			return n1*t*t + 0.984375
		}
	}
)
