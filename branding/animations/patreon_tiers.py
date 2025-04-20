#!/usr/bin/env python3
# =============================================================================
# Patreon Tiers Animation
# =============================================================================
# This script creates an animation for the Patreon tiers.
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
PATREON_BLUE = "#052D49"
PATREON_ORANGE = "#F96854"
TIER1_COLOR = "#6ECACF"  # Teal
TIER2_COLOR = "#9C89B8"  # Purple
TIER3_COLOR = "#F0A202"  # Gold

class PatreonBackground(VMobject):
    def __init__(self, num_particles=40, **kwargs):
        super().__init__(**kwargs)
        self.num_particles = num_particles
        self.create_background()

    def create_background(self):
        # Create floating particles
        particles = VGroup()

        for _ in range(self.num_particles):
            # Random position
            x = random.uniform(-7, 7)
            y = random.uniform(-4, 4)

            # Random size
            size = random.uniform(0.05, 0.2)

            # Random shape (circle or square)
            if random.random() < 0.7:
                particle = Circle(
                    radius=size,
                    fill_color=random.choice([PATREON_ORANGE, TIER1_COLOR, TIER2_COLOR, TIER3_COLOR]),
                    fill_opacity=random.uniform(0.1, 0.3),
                    stroke_width=0
                )
            else:
                particle = Square(
                    side_length=size*2,
                    fill_color=random.choice([PATREON_ORANGE, TIER1_COLOR, TIER2_COLOR, TIER3_COLOR]),
                    fill_opacity=random.uniform(0.1, 0.3),
                    stroke_width=0
                )

            particle.move_to([x, y, 0])
            particles.add(particle)

        self.add(particles)
        self.particles = particles

    def start_floating(self, scene):
        # Create animations for floating particles
        animations = []

        for particle in self.particles:
            # Random movement
            target_x = particle.get_center()[0] + random.uniform(-0.5, 0.5)
            target_y = particle.get_center()[1] + random.uniform(-0.5, 0.5)

            anim = particle.animate.move_to([target_x, target_y, 0])
            animations.append(anim)

        # Play the animation with a slow rate function
        scene.play(
            *animations,
            rate_func=lambda t: np.sin(t * np.pi),
            run_time=10,
            repeat=True
        )

class PatreonLogo(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_logo()

    def create_logo(self):
        # Create the Patreon "P" logo
        circle = Circle(
            radius=0.3,
            fill_color=PATREON_ORANGE,
            fill_opacity=1,
            stroke_width=0
        )

        # Create the vertical bar
        bar = Rectangle(
            height=0.6,
            width=0.15,
            fill_color=PATREON_ORANGE,
            fill_opacity=1,
            stroke_width=0
        )
        bar.next_to(circle, LEFT, buff=-0.15)
        bar.shift(DOWN * 0.15)

        # Create the text
        text = Text("Patreon", font="Arial", weight=BOLD, color=PATREON_ORANGE)
        text.scale(0.8)
        text.next_to(circle, RIGHT, buff=0.2)

        self.add(bar, circle, text)

class TierIcon(VGroup):
    def __init__(self, icon, color, **kwargs):
        super().__init__(**kwargs)
        self.icon = icon
        self.color = color
        self.create_icon()

    def create_icon(self):
        # Create background circle with glow
        bg_glow = Circle(
            radius=0.4,
            fill_color=self.color,
            fill_opacity=0.3,
            stroke_width=0
        )

        bg = Circle(
            radius=0.3,
            fill_color=self.color,
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=1.5
        )

        # Create icon
        icon = Text(self.icon, font="Arial", color=WHITE)
        icon.scale(0.8)
        icon.move_to(bg.get_center())

        self.add(bg_glow, bg, icon)

class BenefitItem(VGroup):
    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)
        self.text = text
        self.create_item()

    def create_item(self):
        # Create checkmark
        checkmark = Text("✔", font="Arial", color=WHITE)
        checkmark.scale(0.4)

        # Create text
        text = Text(self.text, font="Arial", color=WHITE)
        text.scale(0.4)
        text.next_to(checkmark, RIGHT, buff=0.2)

        self.add(checkmark, text)

class PatreonTier(VGroup):
    def __init__(self, name, price, icon, benefits, color, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.price = price
        self.icon = icon
        self.benefits = benefits
        self.color = color
        self.create_tier()

    def create_tier(self):
        # Create the main card with 3D effect
        card_shadow = RoundedRectangle(
            height=5,
            width=3.8,
            corner_radius=0.3,
            fill_color=BLACK,
            fill_opacity=0.5,
            stroke_width=0
        )
        card_shadow.shift(RIGHT * 0.05 + DOWN * 0.05)

        card = RoundedRectangle(
            height=5,
            width=3.8,
            corner_radius=0.3,
            fill_color=self.color,
            fill_opacity=0.9,
            stroke_color=WHITE,
            stroke_width=2
        )

        # Add inner highlight for 3D effect
        highlight = RoundedRectangle(
            height=4.8,
            width=3.6,
            corner_radius=0.25,
            stroke_color=WHITE,
            stroke_width=1,
            stroke_opacity=0.3,
            fill_opacity=0
        )
        highlight.move_to(card.get_center())

        # Create tier icon
        tier_icon = TierIcon(self.icon, self.color)
        tier_icon.scale(0.8)
        tier_icon.next_to(card.get_top(), DOWN, buff=0.4)

        # Create the tier name with 3D effect
        name_shadow = Text(self.name, font="Ubuntu", weight=BOLD, color=BLACK)
        name_shadow.scale(0.6)
        name_shadow.next_to(tier_icon, DOWN, buff=0.2)
        name_shadow.shift(RIGHT * 0.02 + DOWN * 0.02)
        name_shadow.set_opacity(0.5)

        name = Text(self.name, font="Ubuntu", weight=BOLD, color=WHITE)
        name.scale(0.6)
        name.next_to(tier_icon, DOWN, buff=0.2)

        # Create the price with emphasis
        price_bg = RoundedRectangle(
            height=0.5,
            width=2,
            corner_radius=0.1,
            fill_color=WHITE,
            fill_opacity=0.2,
            stroke_color=WHITE,
            stroke_width=1
        )
        price_bg.next_to(name, DOWN, buff=0.2)

        price = Text(self.price, font="Ubuntu", weight=BOLD, color=WHITE)
        price.scale(0.5)
        price.move_to(price_bg.get_center())

        # Create a decorative separator
        separator = Line(
            card.get_left() + RIGHT * 0.5,
            card.get_right() + LEFT * 0.5,
            color=WHITE,
            stroke_width=1
        )
        separator.next_to(price_bg, DOWN, buff=0.3)

        # Add decorative elements to separator
        dot_left = Dot(color=WHITE).move_to(separator.get_start())
        dot_right = Dot(color=WHITE).move_to(separator.get_end())

        # Create the benefits with checkmarks
        benefits_group = VGroup()
        for benefit_text in self.benefits:
            benefit = BenefitItem(benefit_text)
            benefits_group.add(benefit)

        benefits_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        benefits_group.next_to(separator, DOWN, buff=0.4)
        benefits_group.shift(RIGHT * 0.2)  # Adjust alignment

        # Add a subtle pattern background
        pattern = self.create_pattern(card.width - 0.4, card.height - 0.4, self.color)
        pattern.move_to(card.get_center())
        pattern.set_opacity(0.1)

        self.add(card_shadow, card, highlight, pattern, tier_icon, name_shadow, name, price_bg, price, separator, dot_left, dot_right, benefits_group)

    def create_pattern(self, width, height, color):
        pattern = VGroup()

        # Create a grid pattern
        spacing = 0.2
        for x in np.arange(-width/2, width/2, spacing):
            line = Line(
                [x, -height/2, 0],
                [x, height/2, 0],
                stroke_color=WHITE,
                stroke_width=0.5,
                stroke_opacity=0.1
            )
            pattern.add(line)

        for y in np.arange(-height/2, height/2, spacing):
            line = Line(
                [-width/2, y, 0],
                [width/2, y, 0],
                stroke_color=WHITE,
                stroke_width=0.5,
                stroke_opacity=0.1
            )
            pattern.add(line)

        return pattern

class PatreonTiers(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = PATREON_BLUE

        # Create animated background
        background = PatreonBackground(num_particles=30)
        background.set_opacity(0.5)

        # Create the title with glowing effect
        title_text = "Support Stan's ML Stack on Patreon"
        title = Text(title_text, font="Ubuntu", weight=BOLD, color=WHITE)
        title.scale(1.2)
        title.to_edge(UP, buff=0.7)

        # Add glow effect to title
        title_glow = Text(title_text, font="Ubuntu", weight=BOLD, color=PATREON_ORANGE)
        title_glow.scale(1.2)
        title_glow.to_edge(UP, buff=0.7)
        title_glow.set_opacity(0.6)
        title_glow.set_z_index(-1)

        # Create Patreon logo
        patreon_logo = PatreonLogo()
        patreon_logo.next_to(title, DOWN, buff=0.5)

        # Create the tiers with enhanced visuals

        # Tier 1: Coffee Supporter
        tier1 = PatreonTier(
            "Coffee Supporter",
            "$5/month",
            "☕",  # Coffee emoji
            [
                "Access to exclusive posts",
                "Early access to tutorials",
                "Your name in the credits"
            ],
            TIER1_COLOR
        )

        # Tier 2: Code Enthusiast
        tier2 = PatreonTier(
            "Code Enthusiast",
            "$10/month",
            "💻",  # Computer emoji
            [
                "All Coffee Supporter benefits",
                "Access to source code",
                "Monthly Q&A session",
                "Vote on future content"
            ],
            TIER2_COLOR
        )

        # Tier 3: ML Stack Pro
        tier3 = PatreonTier(
            "ML Stack Pro",
            "$25/month",
            "🧠",  # Brain emoji
            [
                "All Code Enthusiast benefits",
                "1-on-1 support sessions",
                "Custom ML Stack configurations",
                "Early access to new features",
                "Priority bug fixes"
            ],
            TIER3_COLOR
        )

        # Arrange tiers horizontally with perspective effect
        tiers = VGroup(tier1, tier2, tier3)
        tiers.arrange(RIGHT, buff=0.8)
        tiers.next_to(patreon_logo, DOWN, buff=1.0)

        # Add slight rotation for perspective
        tier1.rotate(angle=-0.05)
        tier3.rotate(angle=0.05)

        # Create the call to action with button effect
        cta_bg = RoundedRectangle(
            height=0.8,
            width=5,
            corner_radius=0.2,
            fill_color=PATREON_ORANGE,
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=2
        )
        cta_bg.to_edge(DOWN, buff=0.7)

        cta_text = Text("Join at patreon.com/ScooterLacroix", font="Ubuntu", weight=BOLD, color=WHITE)
        cta_text.scale(0.6)
        cta_text.move_to(cta_bg.get_center())

        cta = VGroup(cta_bg, cta_text)

        # Animation sequence with enhanced effects

        # Start with background
        self.add(background)
        background.start_floating(self)

        # Animate title with glow effect
        self.play(
            Write(title, run_time=1.2),
            FadeIn(title_glow, run_time=1.5)
        )

        # Pulse the glow
        self.play(
            title_glow.animate.scale(1.05).set_opacity(0.8),
            rate_func=there_and_back,
            run_time=1
        )

        # Animate Patreon logo
        self.play(FadeIn(patreon_logo, run_time=0.8))

        # Animate tiers appearing with staggered effect and scaling
        for i, tier in enumerate(tiers):
            self.play(
                FadeIn(tier, run_time=0.8),
                tier.animate.scale(1.1).scale(1/1.1),
                run_time=1
            )

        # Highlight each tier with enhanced effects
        for i, tier in enumerate(tiers):
            # Create spotlight effect
            spotlight = tier.copy()
            spotlight.set_fill(opacity=0)
            spotlight.set_stroke(color=WHITE, width=3, opacity=0.8)

            # Highlight animation
            self.play(FadeIn(spotlight, run_time=0.3))

            # Scale up slightly
            self.play(
                tier.animate.scale(1.05),
                run_time=0.5
            )

            # Show some particle effects around the tier
            particles = VGroup()
            for _ in range(10):
                particle = Dot(
                    radius=random.uniform(0.02, 0.05),
                    color=tier.color,
                    fill_opacity=random.uniform(0.5, 1)
                )

                # Position around the tier
                tier_center = tier.get_center()
                offset_x = random.uniform(-2, 2)
                offset_y = random.uniform(-2.5, 2.5)
                particle.move_to([tier_center[0] + offset_x, tier_center[1] + offset_y, 0])

                particles.add(particle)

            # Animate particles appearing and floating
            self.play(FadeIn(particles, run_time=0.3))

            # Float particles around
            particle_animations = []
            for particle in particles:
                # Random movement
                target_x = particle.get_center()[0] + random.uniform(-0.5, 0.5)
                target_y = particle.get_center()[1] + random.uniform(-0.5, 0.5)

                anim = particle.animate.move_to([target_x, target_y, 0])
                particle_animations.append(anim)

            self.play(
                *particle_animations,
                rate_func=there_and_back,
                run_time=1
            )

            # Scale back down
            self.play(
                tier.animate.scale(1/1.05),
                FadeOut(particles),
                FadeOut(spotlight),
                run_time=0.5
            )

        # Show call to action with button effect
        self.play(FadeIn(cta_bg, run_time=0.5))
        self.play(Write(cta_text, run_time=0.8))

        # Add button press effect
        self.play(
            cta_bg.animate.set_opacity(1.0),
            run_time=0.3
        )
        self.play(
            cta_bg.animate.set_opacity(0.8),
            run_time=0.3
        )

        # Final composition with everything
        final_group = VGroup(title, title_glow, patreon_logo, tiers, cta)

        # Scale and center everything
        self.play(
            final_group.animate.scale(0.95).center(),
            run_time=1.5
        )

        self.wait(3)

if __name__ == "__main__":
    # This code will be executed when the script is run directly
    pass
