#!/usr/bin/env python3
# =============================================================================
# ML Stack Architecture Animation
# =============================================================================
# This script creates an animation for the ML Stack architecture.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
#
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

from manim import *
import random
import numpy as np

# Define color scheme for AMD GPUs
AMD_RED = "#ED1C24"
AMD_BLACK = "#000000"
AMD_GRAY = "#58595B"
AMD_WHITE = "#FFFFFF"

# Define layer colors with more appealing gradient
HW_COLOR = "#FF3131"  # Bright red
ROCM_COLOR = "#FF7E27"  # Orange
FRAMEWORK_COLOR = "#FFD700"  # Gold
EXTENSIONS_COLOR = "#4CBB17"  # Green
APPS_COLOR = "#1E90FF"  # Blue

class CircuitBackground(VMobject):
    def __init__(self, width=14, height=8, line_spacing=0.3, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.line_spacing = line_spacing
        self.create_circuit()

    def create_circuit(self):
        # Create horizontal and vertical lines
        lines = VGroup()

        # Horizontal lines
        for y in np.arange(-self.height/2, self.height/2, self.line_spacing):
            # Skip some lines randomly for more organic look
            if random.random() < 0.3:
                continue

            # Determine line length (some shorter, some longer)
            start_x = -self.width/2 + random.uniform(0, self.width/4)
            end_x = self.width/2 - random.uniform(0, self.width/4)

            line = Line(
                start=[start_x, y, 0],
                end=[end_x, y, 0],
                stroke_width=0.5,
                stroke_opacity=random.uniform(0.1, 0.3),
                color=AMD_WHITE
            )
            lines.add(line)

        # Vertical lines
        for x in np.arange(-self.width/2, self.width/2, self.line_spacing):
            # Skip some lines randomly
            if random.random() < 0.3:
                continue

            # Determine line length
            start_y = -self.height/2 + random.uniform(0, self.height/4)
            end_y = self.height/2 - random.uniform(0, self.height/4)

            line = Line(
                start=[x, start_y, 0],
                end=[x, end_y, 0],
                stroke_width=0.5,
                stroke_opacity=random.uniform(0.1, 0.3),
                color=AMD_WHITE
            )
            lines.add(line)

        # Add some circuit nodes
        nodes = VGroup()
        for _ in range(30):
            x = random.uniform(-self.width/2, self.width/2)
            y = random.uniform(-self.height/2, self.height/2)

            # Different node types
            if random.random() < 0.7:
                # Simple dot
                node = Dot(
                    point=[x, y, 0],
                    radius=random.uniform(0.02, 0.05),
                    color=AMD_WHITE,
                    fill_opacity=random.uniform(0.3, 0.7)
                )
            else:
                # Small circle
                node = Circle(
                    radius=random.uniform(0.05, 0.1),
                    stroke_width=0.5,
                    stroke_opacity=random.uniform(0.3, 0.7),
                    stroke_color=AMD_WHITE,
                    fill_opacity=0
                ).move_to([x, y, 0])

            nodes.add(node)

        self.add(lines, nodes)

class DataParticle(VGroup):
    def __init__(self, color=AMD_RED, **kwargs):
        super().__init__(**kwargs)

        # Create the main particle
        particle = Circle(
            radius=0.05,
            fill_color=color,
            fill_opacity=1,
            stroke_width=0,
        )

        # Add glow effect
        glow = Circle(
            radius=0.1,
            fill_color=color,
            fill_opacity=0.3,
            stroke_width=0,
        )

        self.add(glow, particle)

class ComponentIcon(VGroup):
    def __init__(self, name, icon, color=AMD_WHITE, **kwargs):
        super().__init__(**kwargs)

        # Create background circle
        bg = Circle(
            radius=0.25,
            fill_color=color,
            fill_opacity=0.2,
            stroke_color=color,
            stroke_width=1,
            stroke_opacity=0.8
        )

        # Create icon
        icon_text = Text(icon, font="Ubuntu", color=color)
        icon_text.scale(0.4)
        icon_text.move_to(bg.get_center())

        # Create name label
        name_text = Text(name, font="Noto Sans", color=color)
        name_text.scale(0.3)
        name_text.next_to(bg, DOWN, buff=0.1)

        self.add(bg, icon_text, name_text)

class ArchitectureLayer(VGroup):
    def __init__(self, name, color, components, width=10, height=1.5, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.color = color
        self.components = components
        self.width = width
        self.height = height
        self.create_layer()

    def create_layer(self):
        # Create the main rectangle with rounded corners and gradient
        rect = RoundedRectangle(
            height=self.height,
            width=self.width,
            corner_radius=0.2,
            fill_color=self.color,
            fill_opacity=0.8,
            stroke_color=AMD_WHITE,
            stroke_width=1.5
        )

        # Add inner highlight for 3D effect
        highlight = RoundedRectangle(
            height=self.height - 0.1,
            width=self.width - 0.1,
            corner_radius=0.15,
            stroke_color=AMD_WHITE,
            stroke_width=0.5,
            stroke_opacity=0.3,
            fill_opacity=0
        )
        highlight.move_to(rect.get_center())

        # Create the layer name with 3D effect
        layer_name = Text(self.name, font="Ubuntu", weight=BOLD, color=AMD_WHITE)
        layer_name.scale(0.7)
        layer_name.next_to(rect.get_left(), RIGHT, buff=0.5)

        # Add shadow for 3D effect
        layer_name_shadow = Text(self.name, font="Ubuntu", weight=BOLD, color=AMD_BLACK)
        layer_name_shadow.scale(0.7)
        layer_name_shadow.next_to(rect.get_left(), RIGHT, buff=0.5)
        layer_name_shadow.shift(RIGHT * 0.02 + DOWN * 0.02)
        layer_name_shadow.set_opacity(0.5)
        layer_name_shadow.set_z_index(-1)

        # Create the components with icons
        comp_group = VGroup()

        # Define icons for different components
        icons = {
            # Hardware layer
            "AMD Radeon RX 7900 XTX": "🔋",  # Battery (power)
            "AMD Radeon RX 7800 XT": "🔋",
            # ROCm layer
            "ROCm 6.3": "💻",  # Computer
            "HIP": "🔥",  # Fire
            "RCCL": "🔗",  # Link
            "MIOpen": "🔓",  # Unlock
            # Framework layer
            "PyTorch": "🔥",  # Fire
            "ONNX Runtime": "⚡",  # Lightning
            "MIGraphX": "📈",  # Chart
            # Extensions layer
            "Flash Attention": "👁",  # Eye
            "Triton": "🔬",  # Microscope
            "vLLM": "🚀",  # Rocket
            "TensorRT-LLM": "💪",  # Muscle
            # Applications layer
            "Megatron-LM": "🧠",  # Brain
            "RAPIDS": "💡",  # Light bulb
            "Custom Models": "📜"  # Scroll
        }

        # Default icon if not found
        default_icon = "⚙️"  # Gear

        # Arrange components in a grid
        cols = min(3, len(self.components))
        rows = (len(self.components) + cols - 1) // cols

        for i, comp in enumerate(self.components):
            row = i // cols
            col = i % cols

            icon = icons.get(comp, default_icon)
            component = ComponentIcon(comp, icon, color=AMD_WHITE)

            # Position in grid
            x_pos = rect.get_center()[0] + (col - (cols-1)/2) * 1.5
            y_pos = rect.get_center()[1] + (row - (rows-1)/2) * 0.6

            component.move_to([x_pos, y_pos, 0])
            comp_group.add(component)

        # Add circuit pattern inside the rectangle
        circuit = CircuitBackground(width=self.width-0.5, height=self.height-0.3)
        circuit.move_to(rect.get_center())
        circuit.set_opacity(0.15)

        self.rect = rect
        self.add(rect, highlight, circuit, layer_name_shadow, layer_name, comp_group)
        self.original_color = self.color

class DataFlowParticle(VMobject):
    def __init__(self, path, color=AMD_RED, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.color = color
        self.create_particles()

    def create_particles(self):
        self.particles = VGroup()
        for _ in range(5):
            particle = DataParticle(color=self.color)
            particle.scale(random.uniform(0.7, 1.3))
            self.particles.add(particle)

        self.add(self.particles)

    def start_flow(self, scene):
        animations = []
        for i, particle in enumerate(self.particles):
            # Start at different positions
            start_alpha = i * 0.2

            anim = MoveAlongPath(
                particle, self.path,
                rate_func=lambda t: (t + start_alpha) % 1,
                run_time=3,
                repeat=True
            )
            animations.append(anim)

        scene.play(*animations)

class MLStackArchitecture(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = AMD_BLACK

        # Create circuit background
        circuit_bg = CircuitBackground(width=14, height=8)
        circuit_bg.set_opacity(0.2)

        # Create the title with glowing effect
        title_text = "Stan's ML Stack Architecture"
        title = Text(title_text, font="Ubuntu", weight=BOLD, color=AMD_WHITE)
        title.scale(1.2)
        title.to_edge(UP, buff=0.5)

        # Add glow effect to title
        title_glow = Text(title_text, font="Ubuntu", weight=BOLD, color=AMD_RED)
        title_glow.scale(1.2)
        title_glow.to_edge(UP, buff=0.5)
        title_glow.set_opacity(0.6)
        title_glow.set_z_index(-1)

        # Create the layers of the ML Stack with enhanced visuals
        layers = VGroup()

        # Hardware layer
        hw_layer = ArchitectureLayer(
            "Hardware",
            HW_COLOR,
            ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7800 XT"]
        )

        # ROCm layer
        rocm_layer = ArchitectureLayer(
            "ROCm",
            ROCm_COLOR,
            ["ROCm 6.3", "HIP", "RCCL", "MIOpen"]
        )

        # Framework layer
        framework_layer = ArchitectureLayer(
            "Frameworks",
            FRAMEWORK_COLOR,
            ["PyTorch", "ONNX Runtime", "MIGraphX"]
        )

        # Extensions layer
        extensions_layer = ArchitectureLayer(
            "Extensions",
            EXTENSIONS_COLOR,
            ["Flash Attention", "Triton", "vLLM", "TensorRT-LLM"]
        )

        # Applications layer
        apps_layer = ArchitectureLayer(
            "Applications",
            APPS_COLOR,
            ["Megatron-LM", "RAPIDS", "Custom Models"]
        )

        # Stack the layers with spacing
        layers.add(hw_layer, rocm_layer, framework_layer, extensions_layer, apps_layer)
        for i, layer in enumerate(layers):
            if i > 0:
                layer.next_to(layers[i-1], UP, buff=0.6)

        layers.center()

        # Create enhanced arrows between layers
        arrows = VGroup()
        data_flows = []

        for i in range(len(layers) - 1):
            # Create curved path for data flow
            start = layers[i].get_top() + UP * 0.1
            end = layers[i+1].get_bottom() + DOWN * 0.1

            # Create multiple paths for more interesting flow
            paths = []
            for j in range(3):
                offset = (j - 1) * 0.3

                path = CubicBezier(
                    start + RIGHT * offset,
                    start + UP * 0.3 + RIGHT * offset,
                    end + DOWN * 0.3 + RIGHT * offset,
                    end + RIGHT * offset
                )

                # Create visible arrow
                arrow = VMobject()
                arrow.set_points(path.get_points())
                arrow.set_stroke(
                    color=interpolate_color(layers[i].color, layers[i+1].color, 0.5),
                    width=2,
                    opacity=0.8
                )
                arrows.add(arrow)

                # Create data flow
                data_flow = DataFlowParticle(
                    path,
                    color=interpolate_color(layers[i].color, layers[i+1].color, 0.5)
                )
                data_flows.append(data_flow)

        # Create side connections between components
        side_connections = VGroup()
        side_data_flows = []

        # Connect some components between adjacent layers
        for i in range(len(layers) - 1):
            # Get random components from each layer
            if len(layers[i].components) > 0 and len(layers[i+1].components) > 0:
                # Pick random components to connect
                for _ in range(2):
                    comp1 = random.choice(layers[i][5:])  # Skip the rect, highlight, circuit, shadows, and layer name
                    comp2 = random.choice(layers[i+1][5:])  # Skip the rect, highlight, circuit, shadows, and layer name

                    # Create path
                    start = comp1.get_top()
                    end = comp2.get_bottom()

                    path = CubicBezier(
                        start,
                        start + UP * 0.2,
                        end + DOWN * 0.2,
                        end
                    )

                    # Create visible connection
                    connection = VMobject()
                    connection.set_points(path.get_points())
                    connection.set_stroke(
                        color=interpolate_color(layers[i].color, layers[i+1].color, 0.5),
                        width=1,
                        opacity=0.4
                    )
                    side_connections.add(connection)

                    # Create data flow
                    data_flow = DataFlowParticle(
                        path,
                        color=interpolate_color(layers[i].color, layers[i+1].color, 0.5)
                    )
                    side_data_flows.append(data_flow)

        # Create a 3D perspective effect for the entire stack
        stack_group = VGroup(layers, arrows, side_connections)

        # Create the tagline
        tagline = Text("Optimized for AMD Radeon GPUs", font="Ubuntu", color=AMD_WHITE)
        tagline.scale(0.8)
        tagline.to_edge(DOWN, buff=0.5)

        # Add a highlight to "AMD Radeon"
        tagline_parts = [
            Text("Optimized for ", font="Ubuntu", color=AMD_WHITE),
            Text("AMD Radeon", font="Ubuntu", color=AMD_RED),
            Text(" GPUs", font="Ubuntu", color=AMD_WHITE)
        ]

        for i, part in enumerate(tagline_parts):
            part.scale(0.8)
            if i > 0:
                part.next_to(tagline_parts[i-1], RIGHT, buff=0.05)

        tagline_group = VGroup(*tagline_parts)
        tagline_group.move_to(tagline.get_center())

        # Animation sequence with more dynamic effects

        # Start with the circuit background fading in
        self.play(FadeIn(circuit_bg, run_time=1.5))

        # Animate title with glow effect
        self.play(
            Write(title, run_time=1),
            FadeIn(title_glow, run_time=1.5)
        )

        # Pulse the glow
        self.play(
            title_glow.animate.scale(1.05).set_opacity(0.8),
            rate_func=there_and_back,
            run_time=1
        )

        # Animate layers appearing from bottom to top with 3D effect
        for i, layer in enumerate(reversed(layers)):
            self.play(
                FadeIn(layer, run_time=0.8),
                layer.animate.scale(1.05).scale(1/1.05),
                run_time=1
            )

        # Animate main arrows appearing
        for arrow in arrows:
            self.play(Create(arrow, run_time=0.4))

        # Animate side connections
        for connection in side_connections:
            self.play(Create(connection, run_time=0.3))

        # Start data flows
        for data_flow in data_flows:
            self.add(data_flow)
            data_flow.start_flow(self)

        # Start side data flows
        for data_flow in side_data_flows:
            self.add(data_flow)
            data_flow.start_flow(self)

        # Animate tagline appearing
        self.play(Write(tagline_group, run_time=1))

        # Create a 3D rotation effect for the entire architecture
        self.play(
            stack_group.animate.rotate(angle=0.05, axis=UP),
            run_time=1.5
        )

        # Highlight each layer with pulsing effect
        for layer in layers:
            # Create highlight effect
            highlight = layer.rect.copy()
            highlight.set_fill(opacity=0)
            highlight.set_stroke(color=AMD_WHITE, width=3, opacity=0.8)

            self.play(FadeIn(highlight, run_time=0.3))
            self.play(FadeOut(highlight, run_time=0.5))

        # Create a data pulse that travels through the entire stack
        for i in range(len(layers)):
            # Highlight the current layer
            self.play(
                layers[i].animate.set_opacity(1.2),
                run_time=0.4
            )
            self.play(
                layers[i].animate.set_opacity(1.0),
                run_time=0.4
            )

        # Final composition
        final_group = VGroup(
            title, title_glow, layers, arrows, side_connections, tagline_group
        )

        # Scale and center everything
        self.play(
            final_group.animate.scale(0.95).center(),
            run_time=1.5
        )

        self.wait(3)

if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass
