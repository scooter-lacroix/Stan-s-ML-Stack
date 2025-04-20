#!/usr/bin/env python3
# =============================================================================
# GitHub Repository Animation
# =============================================================================
# This script creates an animation for the GitHub repository.
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

# Define color scheme
GITHUB_DARK = "#0D1117"
GITHUB_LIGHT = "#F6F8FA"
GITHUB_BLUE = "#58A6FF"
GITHUB_GREEN = "#2EA043"
GITHUB_PURPLE = "#8957E5"
GITHUB_ORANGE = "#F97583"
GITHUB_RED = "#F85149"
GITHUB_YELLOW = "#E3B341"

# Additional colors for variety
AMD_RED = "#ED1C24"

# Base Classes:
# 1. GitHubBackground - Code pattern background
# 2. CodeBlock - Animated code snippet display
# 3. RepositoryStructure - Enhanced file tree visualization
# 4. FeatureCard - Card with icon and description
# 5. StarAnimation - Star burst animation

# Main Scene Structure:
# - Background setup with code patterns
# - Title with glow effect
# - GitHub logo with 3D effect
# - Repository structure visualization
# - Feature cards with icons
# - Statistics display (stars, forks)
# - Call to action

# Animation Sequence:
# - Background fade in
# - Title animation with glow
# - GitHub logo appearance
# - Repository structure building
# - Feature cards appearing with icons
# - Statistics counter animation
# - Call to action with button effect

class GitHubRepo(Scene):
    def construct(self):
        # Implementation will go here
        pass

if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass

# Base Classes Implementation

class GitHubBackground(VMobject):
    """Creates a code pattern background for GitHub."""
    def __init__(self, width=14, height=8, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.create_background()

    def create_background(self):
        # Create code-like pattern
        code_lines = VGroup()

        # Create horizontal lines of varying length to simulate code
        num_lines = 40
        for i in range(num_lines):
            # Determine line position
            y_pos = -self.height/2 + i * (self.height / num_lines)

            # Randomize line length to simulate code
            line_length = random.uniform(0.5, self.width - 1)
            start_x = -self.width/2 + random.uniform(0, 1)

            # Create line
            line = Line(
                start=[start_x, y_pos, 0],
                end=[start_x + line_length, y_pos, 0],
                stroke_width=1,
                stroke_opacity=random.uniform(0.1, 0.3),
                color=random.choice([GITHUB_LIGHT, GITHUB_BLUE, GITHUB_GREEN, GITHUB_PURPLE])
            )
            code_lines.add(line)

        # Add some "brackets" and "parentheses"
        brackets = VGroup()
        for _ in range(15):
            x = random.uniform(-self.width/2 + 1, self.width/2 - 1)
            y = random.uniform(-self.height/2 + 1, self.height/2 - 1)
            height = random.uniform(0.2, 0.8)

            # Choose bracket type
            bracket_type = random.choice(["()", "[]", "{}"])

            if bracket_type == "()":
                left_bracket = Arc(
                    angle=PI,
                    start_angle=PI/2,
                    radius=height/2,
                    stroke_width=1,
                    stroke_opacity=random.uniform(0.2, 0.4),
                    color=GITHUB_LIGHT
                )
                right_bracket = Arc(
                    angle=PI,
                    start_angle=-PI/2,
                    radius=height/2,
                    stroke_width=1,
                    stroke_opacity=random.uniform(0.2, 0.4),
                    color=GITHUB_LIGHT
                )
                right_bracket.next_to(left_bracket, RIGHT, buff=height)
            elif bracket_type == "[]":
                left_bracket = Line(
                    [0, height/2, 0],
                    [0, -height/2, 0],
                    stroke_width=1,
                    stroke_opacity=random.uniform(0.2, 0.4),
                    color=GITHUB_LIGHT
                )
                right_bracket = Line(
                    [0, height/2, 0],
                    [0, -height/2, 0],
                    stroke_width=1,
                    stroke_opacity=random.uniform(0.2, 0.4),
                    color=GITHUB_LIGHT
                )
                right_bracket.next_to(left_bracket, RIGHT, buff=height)
            else:  # "{}"
                left_bracket = VMobject()
                left_bracket.set_points_as_corners([
                    [0, height/2, 0],
                    [-0.1, height/2, 0],
                    [-0.1, -height/2, 0],
                    [0, -height/2, 0]
                ])
                left_bracket.set_stroke(
                    color=GITHUB_LIGHT,
                    width=1,
                    opacity=random.uniform(0.2, 0.4)
                )

                right_bracket = VMobject()
                right_bracket.set_points_as_corners([
                    [0, height/2, 0],
                    [0.1, height/2, 0],
                    [0.1, -height/2, 0],
                    [0, -height/2, 0]
                ])
                right_bracket.set_stroke(
                    color=GITHUB_LIGHT,
                    width=1,
                    opacity=random.uniform(0.2, 0.4)
                )
                right_bracket.next_to(left_bracket, RIGHT, buff=height)

            bracket_group = VGroup(left_bracket, right_bracket)
            bracket_group.move_to([x, y, 0])
            brackets.add(bracket_group)

        self.add(code_lines, brackets)

class CodeBlock(VGroup):
    """Creates an animated code snippet display."""
    def __init__(self, code_text, language="python", **kwargs):
        super().__init__(**kwargs)
        self.code_text = code_text
        self.language = language
        self.create_code_block()

    def create_code_block(self):
        # Create background
        background = Rectangle(
            width=6,
            height=3,
            fill_color=GITHUB_DARK,
            fill_opacity=1,
            stroke_color=GITHUB_LIGHT,
            stroke_width=1,
            stroke_opacity=0.3
        )

        # Create language indicator
        language_bg = Rectangle(
            width=1.5,
            height=0.4,
            fill_color=GITHUB_PURPLE,
            fill_opacity=0.8,
            stroke_width=0
        )
        language_bg.next_to(background.get_top(), DOWN, buff=0.1)
        language_bg.shift(LEFT * 2)

        language_text = Text(self.language, font="Noto Sans", color=GITHUB_LIGHT)
        language_text.scale(0.3)
        language_text.move_to(language_bg.get_center())

        # Create code text with syntax highlighting
        code_lines = self.code_text.strip().split("\n")
        code_group = VGroup()

        for i, line in enumerate(code_lines):
            # Apply simple syntax highlighting
            highlighted_line = self.highlight_syntax(line)
            # Position each line relative to the background
            highlighted_line.move_to(background.get_center())
            highlighted_line.shift(UP * (1 - i * 0.4) + LEFT * 2.5)
            code_group.add(highlighted_line)

        # Create line numbers
        line_numbers = VGroup()
        for i in range(len(code_lines)):
            number = Text(str(i+1), font="Noto Sans", color=GITHUB_LIGHT)
            number.scale(0.25)
            number.next_to(code_group[i], LEFT, buff=0.2)
            line_numbers.add(number)

        self.add(background, language_bg, language_text, line_numbers, code_group)
        self.background = background
        self.code_group = code_group

    def highlight_syntax(self, line):
        """Apply syntax highlighting to a line of code."""
        # This is a simplified version - a real implementation would use regex
        # to properly parse and highlight code

        # Handle empty lines
        if not line.strip():
            empty_text = Text(" ", font="Noto Sans", color=GITHUB_LIGHT)
            empty_text.scale(0.3)
            return empty_text

        # Split the line into parts that can be colored differently
        parts = []
        current_part = ""
        current_color = GITHUB_LIGHT  # Default color

        # Simple parsing for Python-like syntax
        for char in line:
            if char in "()[]{}":
                if current_part:
                    parts.append((current_part, current_color))
                    current_part = ""
                parts.append((char, GITHUB_ORANGE))
            elif char in "\"'":
                if current_part:
                    parts.append((current_part, current_color))
                    current_part = ""
                parts.append((char, GITHUB_GREEN))
            else:
                current_part += char

        if current_part:
            parts.append((current_part, current_color))

        # Create text objects for each part
        text_parts = VGroup()

        # Handle case where parts might be empty
        if not parts:
            empty_text = Text(" ", font="Noto Sans", color=GITHUB_LIGHT)
            empty_text.scale(0.3)
            return empty_text

        for part_text, part_color in parts:
            part = Text(part_text, font="Noto Sans", color=part_color)
            part.scale(0.3)
            part.next_to(text_parts, RIGHT, buff=0) if text_parts else None
            text_parts.add(part)

        return text_parts

class RepositoryStructure(VGroup):
    """Creates an enhanced file tree visualization."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_structure()

    def create_structure(self):
        # Create background
        background = RoundedRectangle(
            width=5,
            height=6,
            corner_radius=0.2,
            fill_color=GITHUB_DARK,
            fill_opacity=0.8,
            stroke_color=GITHUB_LIGHT,
            stroke_width=1
        )

        # Create title
        title = Text("Repository Structure", font="Noto Sans", color=GITHUB_LIGHT)
        title.scale(0.5)
        title.next_to(background.get_top(), DOWN, buff=0.3)

        # Create file tree
        file_tree = self.create_file_tree()
        file_tree.next_to(title, DOWN, buff=0.4)

        self.add(background, title, file_tree)
        self.background = background
        self.file_tree = file_tree

    def create_file_tree(self):
        """Create the file tree structure."""
        # Define the structure
        structure = [
            ("📁 core", GITHUB_BLUE, [
                ("  📁 pytorch", GITHUB_BLUE),
                ("  📁 onnx", GITHUB_BLUE),
                ("  📁 migraphx", GITHUB_BLUE),
                ("  📁 megatron", GITHUB_BLUE),
                ("  📁 flash_attention", GITHUB_BLUE)
            ]),
            ("📁 extensions", GITHUB_PURPLE, [
                ("  📁 triton", GITHUB_PURPLE),
                ("  📁 vllm", GITHUB_PURPLE),
                ("  📁 tensorrt_llm", GITHUB_PURPLE)
            ]),
            ("📁 docs", GITHUB_GREEN, [
                ("  📄 README.md", GITHUB_LIGHT),
                ("  📄 INSTALLATION.md", GITHUB_LIGHT)
            ]),
            ("📁 tests", GITHUB_ORANGE, [
                ("  📄 test_gpu.py", GITHUB_LIGHT),
                ("  📄 test_flash_attn.py", GITHUB_LIGHT)
            ]),
            ("📁 benchmarks", GITHUB_YELLOW, [
                ("  📄 benchmark_gpu.py", GITHUB_LIGHT)
            ]),
            ("📄 README.md", GITHUB_LIGHT),
            ("📄 LICENSE", GITHUB_LIGHT)
        ]

        # Create the tree
        tree = VGroup()

        # Track the current column and row
        current_row = 0
        current_column = 0
        max_rows_per_column = 4  # Adjust this to control when to start a new column

        for i, (item, color, *children) in enumerate(structure):
            # Create item
            item_text = Text(item, font="Noto Sans", color=color)
            item_text.scale(0.4)

            # Position based on current row and column
            if i == 0:
                # First item
                item_text.to_corner(UL, buff=0.3)
                item_text.shift(DOWN * 0.5 + RIGHT * 0.5)
            else:
                if current_row < max_rows_per_column:
                    # Continue in the same column
                    item_text.next_to(tree[-1], DOWN, buff=0.2)
                    item_text.align_to(tree[current_column * max_rows_per_column], LEFT)
                else:
                    # Start a new column
                    current_column += 1
                    current_row = 0
                    reference_item = tree[0]  # First item in the tree
                    item_text.next_to(reference_item, RIGHT, buff=2.5 * current_column)
                    item_text.align_to(reference_item, UP)

            tree.add(item_text)

            # Add children if any
            if children:
                for child_item, child_color in children[0]:
                    child_text = Text(child_item, font="Noto Sans", color=child_color)
                    child_text.scale(0.35)
                    child_text.next_to(tree[-1], DOWN, buff=0.15)
                    child_text.align_to(tree[-1], LEFT)
                    child_text.shift(RIGHT * 0.3)  # Indent

                    tree.add(child_text)
                    current_row += 1

            current_row += 1

        return tree

class FeatureCard(VGroup):
    """Creates a card with icon and description."""
    def __init__(self, title, description, icon, color=GITHUB_BLUE, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.description = description
        self.icon = icon
        self.color = color
        self.create_card()

    def create_card(self):
        # Create card background with 3D effect
        card_shadow = RoundedRectangle(
            width=4.5,  # Wider card
            height=2.2,  # Taller card
            corner_radius=0.2,
            fill_color=BLACK,
            fill_opacity=0.3,
            stroke_width=0
        )
        card_shadow.shift(RIGHT * 0.03 + DOWN * 0.03)

        card = RoundedRectangle(
            width=4.5,  # Wider card
            height=2.2,  # Taller card
            corner_radius=0.2,
            fill_color=GITHUB_DARK,
            fill_opacity=0.8,
            stroke_color=self.color,
            stroke_width=1.5
        )

        # Create icon
        icon = Text(self.icon, font="Noto Sans", color=self.color)
        icon.scale(0.8)
        icon.move_to(card.get_left() + RIGHT * 0.8 + UP * 0.5)

        # Create title
        title = Text(self.title, font="Noto Sans", weight=BOLD, color=GITHUB_LIGHT)
        title.scale(0.4)
        title.next_to(icon, RIGHT, buff=0.3)
        title.align_to(icon, UP)  # Align to top of icon

        # Create description
        description = Text(self.description, font="Noto Sans", color=GITHUB_LIGHT)
        description.scale(0.3)
        description.move_to(card.get_center() + DOWN * 0.3)

        # Make sure description doesn't overflow
        if description.width > card.width - 0.5:
            description.scale_to_fit_width(card.width - 0.5)

        self.add(card_shadow, card, icon, title, description)

class StarAnimation(VMobject):
    """Creates a star burst animation."""
    def __init__(self, color=GITHUB_YELLOW, **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.create_star()

    def create_star(self):
        # Create star shape
        star = Star(
            n=5,
            outer_radius=0.3,
            inner_radius=0.15,
            fill_color=self.color,
            fill_opacity=1,
            stroke_color=GITHUB_LIGHT,
            stroke_width=2
        )

        # Create glow effect
        glow = Star(
            n=5,
            outer_radius=0.4,
            inner_radius=0.2,
            fill_color=self.color,
            fill_opacity=0.3,
            stroke_width=0
        )

        self.add(glow, star)
        self.star = star
        self.glow = glow

    def animate_star(self, scene):
        """Animate the star with a burst effect."""
        # Scale up and down
        scene.play(
            self.glow.animate.scale(1.5).set_opacity(0.5),
            self.star.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.5
        )

        # Rotate slightly
        scene.play(
            Rotate(self.star, angle=0.3),
            rate_func=there_and_back,
            run_time=0.3
        )

# Main Scene Implementation

class GitHubRepo(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = GITHUB_DARK

        # Create code background
        code_bg = GitHubBackground(width=14, height=8)
        code_bg.set_opacity(0.2)

        # Create the title with glowing effect
        title_text = "Stan's ML Stack on GitHub"
        title = Text(title_text, font="Noto Sans", weight=BOLD, color=GITHUB_LIGHT)
        title.scale(1.2)
        title.to_edge(UP, buff=0.5)

        # Add glow effect to title
        title_glow = Text(title_text, font="Noto Sans", weight=BOLD, color=GITHUB_BLUE)
        title_glow.scale(1.2)
        title_glow.to_edge(UP, buff=0.5)
        title_glow.set_opacity(0.6)
        title_glow.set_z_index(-1)

        # Create GitHub logo with 3D effect
        github_logo = self.create_github_logo()
        github_logo.next_to(title, DOWN, buff=0.5)

        # Create repository structure
        repo_structure = RepositoryStructure()
        repo_structure.scale(0.9)
        repo_structure.to_edge(LEFT, buff=1)
        repo_structure.shift(DOWN * 0.5)

        # Create feature cards
        features = self.create_features()
        features.scale(0.9)
        features.to_edge(RIGHT, buff=1)
        features.shift(DOWN * 0.5)

        # Create statistics display
        stats = self.create_stats()
        stats.next_to(repo_structure, DOWN, buff=0.5)
        stats.align_to(repo_structure, LEFT)

        # Create the call to action
        cta = self.create_cta()
        cta.to_edge(DOWN, buff=0.7)

        # Create sample code block
        code_sample = self.create_code_sample()
        code_sample.scale(0.8)
        code_sample.next_to(features, DOWN, buff=0.5)
        code_sample.align_to(features, RIGHT)

        # Animation sequence

        # Start with the code background fading in
        self.play(FadeIn(code_bg, run_time=1.5))

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

        # Animate GitHub logo
        self.play(FadeIn(github_logo, run_time=0.8))

        # Animate repository structure appearing
        self.play(FadeIn(repo_structure.background, run_time=0.5))

        # Animate file tree items appearing one by one
        for item in repo_structure.file_tree:
            self.play(FadeIn(item, run_time=0.1))

        # Animate feature cards appearing
        for feature in features:
            self.play(
                FadeIn(feature, run_time=0.5),
                feature.animate.scale(1.05).scale(1/1.05),
                run_time=0.7
            )

        # Animate code sample appearing
        self.play(FadeIn(code_sample, run_time=0.8))

        # Animate typing effect for code
        for line in code_sample.code_group:
            self.play(Write(line, run_time=0.3))

        # Animate statistics appearing with counter effect
        self.play(FadeIn(stats[0], run_time=0.5))  # Stats background

        # Animate star counter
        star_count = 0
        target_stars = 250
        star_counter = stats[1]

        self.add(star_counter)

        # Create star animation
        star_anim = StarAnimation()
        star_anim.next_to(star_counter, LEFT, buff=0.2)
        self.play(FadeIn(star_anim, run_time=0.3))

        # Animate star counter
        for _ in range(5):  # Just do 5 steps to keep animation short
            increment = target_stars // 5
            star_count += increment
            new_counter = Text(f"{star_count}+ Stars", font="Noto Sans", color=GITHUB_LIGHT)
            new_counter.scale(0.4)
            new_counter.move_to(star_counter.get_center())

            self.play(
                Transform(star_counter, new_counter),
                run_time=0.3
            )

        # Animate star burst
        star_anim.animate_star(self)

        # Animate fork counter
        fork_count = 0
        target_forks = 50
        fork_counter = stats[2]

        self.add(fork_counter)

        # Animate fork counter
        for _ in range(5):  # Just do 5 steps to keep animation short
            increment = target_forks // 5
            fork_count += increment
            new_counter = Text(f"{fork_count}+ Forks", font="Noto Sans", color=GITHUB_LIGHT)
            new_counter.scale(0.4)
            new_counter.move_to(fork_counter.get_center())

            self.play(
                Transform(fork_counter, new_counter),
                run_time=0.3
            )

        # Show call to action with button effect
        self.play(FadeIn(cta, run_time=0.8))

        # Add button press effect
        self.play(
            cta[0].animate.set_opacity(1.0),
            run_time=0.3
        )
        self.play(
            cta[0].animate.set_opacity(0.8),
            run_time=0.3
        )

        # Final composition - scale everything to fit in the frame
        final_group = VGroup(
            title, title_glow, github_logo, repo_structure,
            features, code_sample, cta, stats
        )

        # Scale and center everything
        self.play(
            final_group.animate.scale(0.85).center(),
            run_time=1.5
        )

        self.wait(3)

    def create_github_logo(self):
        """Create GitHub logo with 3D effect."""
        # Create the GitHub octocat silhouette (simplified)
        octocat = SVGMobject("assets/github_octocat.svg")
        octocat.set_color(GITHUB_LIGHT)
        octocat.set_stroke(width=0)

        # Add shadow for 3D effect
        octocat_shadow = octocat.copy()
        octocat_shadow.set_color(BLACK)
        octocat_shadow.set_opacity(0.5)
        octocat_shadow.shift(RIGHT * 0.05 + DOWN * 0.05)
        octocat_shadow.set_z_index(-1)

        # Create text
        github_text = Text("GitHub", font="Noto Sans", weight=BOLD, color=GITHUB_LIGHT)
        github_text.scale(0.8)
        github_text.next_to(octocat, RIGHT, buff=0.3)

        # Group everything
        logo = VGroup(octocat_shadow, octocat, github_text)
        return logo

    def create_features(self):
        """Create feature cards."""
        features = VGroup()

        # Define features
        feature_data = [
            ("AMD GPU Optimized", "Specially tuned for AMD Radeon GPUs", "🔥", AMD_RED),
            ("ROCm Compatible", "Works with ROCm 6.3 and HIP", "⚡", GITHUB_BLUE),
            ("Flash Attention", "Optimized attention mechanism", "👁", GITHUB_PURPLE),
            ("Distributed Training", "Scale across multiple GPUs", "🚀", GITHUB_GREEN),
            ("Python API", "Easy to use Python interface", "🐍", GITHUB_YELLOW)
        ]

        # Create feature cards
        for title, desc, icon, color in feature_data:
            feature = FeatureCard(title, desc, icon, color)
            features.add(feature)

        # Arrange features in a grid with more space
        features.arrange_in_grid(rows=3, cols=2, buff=0.5)

        return features

    def create_stats(self):
        """Create statistics display."""
        # Create background
        background = RoundedRectangle(
            width=4,
            height=1,
            corner_radius=0.2,
            fill_color=GITHUB_DARK,
            fill_opacity=0.7,
            stroke_color=GITHUB_LIGHT,
            stroke_width=1,
            stroke_opacity=0.5
        )

        # Create star counter
        star_counter = Text("0+ Stars", font="Noto Sans", color=GITHUB_LIGHT)
        star_counter.scale(0.4)
        star_counter.move_to(background.get_center() + LEFT * 1)

        # Create fork counter
        fork_counter = Text("0+ Forks", font="Noto Sans", color=GITHUB_LIGHT)
        fork_counter.scale(0.4)
        fork_counter.move_to(background.get_center() + RIGHT * 1)

        return VGroup(background, star_counter, fork_counter)

    def create_cta(self):
        """Create call to action button."""
        # Create button
        button = RoundedRectangle(
            width=5,
            height=0.8,
            corner_radius=0.2,
            fill_color=GITHUB_GREEN,
            fill_opacity=0.8,
            stroke_color=GITHUB_LIGHT,
            stroke_width=1.5
        )

        # Create text
        text = Text("Star the repo at github.com/scooter-lacroix", font="Noto Sans", weight=BOLD, color=GITHUB_LIGHT)
        text.scale(0.4)
        text.move_to(button.get_center())

        return VGroup(button, text)

    def create_code_sample(self):
        """Create a sample code snippet."""
        code_text = """import torch
from stans_mlstack import flash_attention

# Initialize model with Flash Attention
model = flash_attention.FlashAttentionModel(
    hidden_size=1024,
    num_heads=16
)

# Move to AMD GPU
model = model.to("cuda")

# Run inference
output = model(input_tensor)"""

        return CodeBlock(code_text, language="python")
