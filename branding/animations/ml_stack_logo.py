#!/usr/bin/env python3
# =============================================================================
# ML Stack Logo Animation
# =============================================================================
# This script creates an animation for the ML Stack logo.
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

# Define additional colors
BLUE_ACCENT = "#00A4EF"
GREEN_ACCENT = "#7FBA00"
YELLOW_ACCENT = "#FFB900"
PURPLE_ACCENT = "#B4A7D6"

class NeuralNetworkBackground(VMobject):
    def __init__(self, num_nodes=50, num_connections=80, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.num_connections = num_connections
        self.create_network()

    def create_network(self):
        # Create nodes
        nodes = VGroup()
        node_positions = []

        for _ in range(self.num_nodes):
            x = random.uniform(-7, 7)
            y = random.uniform(-4, 4)
            node = Dot(point=[x, y, 0], radius=0.03, color=AMD_WHITE, fill_opacity=random.uniform(0.3, 0.8))
            nodes.add(node)
            node_positions.append((x, y))

        # Create connections
        connections = VGroup()
        for _ in range(self.num_connections):
            i, j = random.sample(range(self.num_nodes), 2)
            start = node_positions[i]
            end = node_positions[j]

            # Calculate distance
            dist = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

            # Only connect if distance is reasonable
            if dist < 3:
                line = Line(
                    [start[0], start[1], 0],
                    [end[0], end[1], 0],
                    stroke_width=0.5,
                    stroke_opacity=random.uniform(0.1, 0.4),
                    color=random.choice([BLUE_ACCENT, GREEN_ACCENT, YELLOW_ACCENT, PURPLE_ACCENT, AMD_RED])
                )
                connections.add(line)

        self.add(connections, nodes)

class GPUChip(VGroup):
    def __init__(self, width=2, height=0.5, color=AMD_RED, **kwargs):
        super().__init__(**kwargs)

        # Main GPU body
        self.body = RoundedRectangle(
            height=height,
            width=width,
            corner_radius=0.1,
            fill_opacity=1,
            fill_color=color,
            stroke_color=AMD_WHITE,
            stroke_width=2
        )

        # Add circuit patterns
        self.circuits = VGroup()

        # Horizontal lines
        for i in range(3):
            y_pos = -height/4 + i * height/3
            line = Line(
                start=[-width/2 + 0.2, y_pos, 0],
                end=[width/2 - 0.2, y_pos, 0],
                stroke_width=0.5,
                stroke_color=AMD_WHITE,
                stroke_opacity=0.5
            )
            self.circuits.add(line)

        # Vertical lines
        for i in range(5):
            x_pos = -width/2 + 0.3 + i * width/5
            line = Line(
                start=[x_pos, -height/2 + 0.1, 0],
                end=[x_pos, height/2 - 0.1, 0],
                stroke_width=0.5,
                stroke_color=AMD_WHITE,
                stroke_opacity=0.5
            )
            self.circuits.add(line)

        # Add connection points
        self.connectors = VGroup()
        for i in range(4):
            x_pos = -width/2 + 0.4 + i * width/4
            connector = Circle(
                radius=0.03,
                fill_opacity=1,
                fill_color=AMD_WHITE,
                stroke_width=0
            ).move_to([x_pos, 0, 0])
            self.connectors.add(connector)

        self.add(self.body, self.circuits, self.connectors)

class DataFlow(VMobject):
    def __init__(self, path, color=AMD_RED, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.color = color
        self.create_flow()

    def create_flow(self):
        # Create particles that will flow along the path
        self.particles = VGroup()
        for _ in range(5):
            particle = Circle(
                radius=0.03,
                fill_opacity=1,
                fill_color=self.color,
                stroke_width=0,
                stroke_opacity=0
            )
            self.particles.add(particle)

        self.add(self.particles)

    def start_flow(self, scene):
        # Animate particles flowing along the path
        animations = []
        for i, particle in enumerate(self.particles):
            # Start at different positions along the path
            start_alpha = i * 0.2

            # Create the motion animation
            anim = MoveAlongPath(
                particle, self.path,
                rate_func=lambda t: (t + start_alpha) % 1,
                run_time=3,
                repeat=True
            )
            animations.append(anim)

        scene.play(*animations)

class MLStackLogo(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = AMD_BLACK

        # Create neural network background
        network_bg = NeuralNetworkBackground(num_nodes=40, num_connections=60)
        network_bg.set_opacity(0.3)

        # Create the title with glowing effect
        title_text = "Stan's ML Stack"
        title = Text(title_text, font="Ubuntu", color=AMD_WHITE)
        title.scale(1.5)
        title.to_edge(UP, buff=0.8)

        # Add glow effect to title
        title_glow = Text(title_text, font="Ubuntu", color=AMD_RED)
        title_glow.scale(1.5)
        title_glow.to_edge(UP, buff=0.8)
        title_glow.set_opacity(0.6)
        title_glow.set_z_index(-1)

        # Create the AMD logo with 3D effect
        amd_logo = Text("AMD", font="Ubuntu", weight=BOLD)
        amd_logo.scale(1.2)
        amd_logo.set_color(AMD_RED)
        amd_logo.next_to(title, DOWN, buff=0.5)

        # Add shadow to AMD logo for 3D effect
        amd_shadow = Text("AMD", font="Ubuntu", weight=BOLD)
        amd_shadow.scale(1.2)
        amd_shadow.set_color(AMD_BLACK)
        amd_shadow.set_opacity(0.5)
        amd_shadow.next_to(title, DOWN, buff=0.5)
        amd_shadow.shift(RIGHT * 0.03 + DOWN * 0.03)
        amd_shadow.set_z_index(-1)

        # Create the GPU stack with enhanced visuals
        gpu_stack = VGroup()
        for i in range(3):
            gpu = GPUChip(
                width=2.5,
                height=0.6,
                color=AMD_RED if i == 0 else AMD_GRAY
            )
            gpu.shift(DOWN * i * 0.7)
            gpu_stack.add(gpu)

        gpu_stack.next_to(amd_logo, DOWN, buff=0.7)

        # Create the ML frameworks with enhanced visuals
        frameworks = VGroup()
        framework_names = ["PyTorch", "ONNX", "MIGraphX", "Megatron-LM"]
        framework_colors = [BLUE_ACCENT, GREEN_ACCENT, YELLOW_ACCENT, PURPLE_ACCENT]
        framework_icons = ["🔥", "⚡", "🚀", "🧠"]  # Emoji icons for each framework

        for i, (name, color, icon) in enumerate(zip(framework_names, framework_colors, framework_icons)):
            # Create framework box
            box = RoundedRectangle(
                height=0.5,
                width=2.5,
                corner_radius=0.1,
                fill_opacity=0.8,
                fill_color=color,
                stroke_color=AMD_WHITE,
                stroke_width=1.5
            )

            # Create framework text
            text = Text(name, font="Ubuntu", color=AMD_WHITE)
            text.scale(0.5)
            text.move_to(box.get_center())

            # Create icon
            icon_text = Text(icon, font="Arial")
            icon_text.scale(0.4)
            icon_text.next_to(text, LEFT, buff=0.2)

            # Group them
            framework = VGroup(box, text, icon_text)
            framework.shift(RIGHT * 3.5 + DOWN * (i * 0.7 - 0.7))
            frameworks.add(framework)

        # Create connecting lines with data flow
        lines = VGroup()
        data_flows = []

        for i, framework in enumerate(frameworks):
            start_point = gpu_stack[min(i, len(gpu_stack) - 1)].get_right()
            end_point = framework[0].get_left()  # Get the box's left point

            # Create curved path for data flow
            control_point = [start_point[0] + (end_point[0] - start_point[0])/2,
                            start_point[1] + (end_point[1] - start_point[1])/2 + random.uniform(-0.2, 0.2),
                            0]

            path = CubicBezier(
                start_point,
                [start_point[0] + 0.5, start_point[1], 0],
                [end_point[0] - 0.5, end_point[1], 0],
                end_point
            )

            # Create the visible line
            line = VMobject()
            line.set_points(path.get_points())
            line.set_stroke(framework_colors[i], width=2, opacity=0.8)
            lines.add(line)

            # Create data flow animation
            data_flow = DataFlow(path, color=framework_colors[i])
            data_flows.append(data_flow)

        # Create the tagline with dynamic effect
        tagline = Text("Optimized for AMD Radeon GPUs", font="Ubuntu", color=AMD_WHITE)
        tagline.scale(0.8)
        tagline.to_edge(DOWN, buff=0.7)

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

        # Start with the neural network background fading in
        self.play(FadeIn(network_bg, run_time=1.5))

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

        # Animate AMD logo with 3D effect
        self.play(
            FadeIn(amd_shadow, run_time=0.5),
            FadeIn(amd_logo, run_time=0.8)
        )

        # Animate GPU stack appearing with staggered effect
        for i, gpu in enumerate(gpu_stack):
            self.play(
                FadeIn(gpu, run_time=0.4),
                gpu.animate.shift(RIGHT * 0.1).shift(LEFT * 0.1),
                run_time=0.5
            )

        # Animate frameworks appearing with staggered effect
        for framework in frameworks:
            self.play(
                FadeIn(framework, run_time=0.4),
                framework.animate.scale(1.1).scale(1/1.1),
                run_time=0.6
            )

        # Animate connecting lines appearing
        for line in lines:
            self.play(Create(line, run_time=0.4))

        # Start data flow animations
        for data_flow in data_flows:
            self.add(data_flow)
            data_flow.start_flow(self)

        # Animate tagline appearing
        self.play(Write(tagline_group, run_time=1))

        # Highlight the main GPU with pulsing effect
        for _ in range(2):
            self.play(
                gpu_stack[0].animate.set_opacity(0.7),
                rate_func=there_and_back,
                run_time=0.7
            )

        # Add some particle effects around the main GPU
        particles = VGroup()
        for _ in range(15):
            particle = Dot(
                radius=random.uniform(0.01, 0.03),
                color=AMD_RED,
                fill_opacity=random.uniform(0.5, 1)
            )

            # Position around the main GPU
            gpu_center = gpu_stack[0].get_center()
            offset_x = random.uniform(-1.5, 1.5)
            offset_y = random.uniform(-0.4, 0.4)
            particle.move_to([gpu_center[0] + offset_x, gpu_center[1] + offset_y, 0])

            particles.add(particle)

        # Animate particles appearing and floating
        self.play(FadeIn(particles, run_time=0.5))

        # Float particles around
        particle_animations = []
        for particle in particles:
            # Random movement
            target_x = particle.get_center()[0] + random.uniform(-0.5, 0.5)
            target_y = particle.get_center()[1] + random.uniform(-0.3, 0.3)

            anim = particle.animate.move_to([target_x, target_y, 0])
            particle_animations.append(anim)

        self.play(
            *particle_animations,
            rate_func=there_and_back,
            run_time=2
        )

        # Final composition with everything
        final_group = VGroup(
            title, title_glow, amd_logo, amd_shadow,
            gpu_stack, frameworks, lines, tagline_group, particles
        )

        # Scale and center everything
        self.play(
            final_group.animate.scale(0.9).center(),
            run_time=1.5
        )

        self.wait(3)

if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass
