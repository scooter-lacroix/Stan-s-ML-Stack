# Stan's ML Stack Branding

This directory contains branding assets and animations for Stan's ML Stack.

## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

## Directory Structure

```
branding/
├── animations/            # Manim animations
│   ├── ml_stack_logo.py   # ML Stack logo animation
│   ├── ml_stack_architecture.py  # ML Stack architecture animation
│   ├── patreon_tiers.py   # Patreon tiers animation
│   ├── github_repo.py     # GitHub repository animation
│   └── run_animations.py  # Script to run all animations
├── images/                # Generated images
│   ├── logo/              # Logo variations
│   ├── banners/           # Banners for GitHub, social media, etc.
│   ├── patreon/           # Patreon tier images
│   └── icons/             # Icons for documentation
└── image_prompts.md       # Prompts for image generation
```

## Animations

The animations are created using [Manim](https://www.manim.community/), a mathematical animation engine. To run the animations, you need to have Manim installed.

### Installation

```bash
# Create a virtual environment
python3 -m venv manim_env

# Activate the virtual environment
source manim_env/bin/activate

# Install Manim
pip install manim

# Install system dependencies
sudo apt-get install ffmpeg libcairo2-dev libpango1.0-dev
```

### Running Animations

To run all animations:

```bash
cd animations
python run_animations.py
```

To run a specific animation:

```bash
cd animations
python -m manim -pqh ml_stack_logo.py MLStackLogo
```

## Image Generation

The `image_prompts.md` file contains prompts for generating images using AI image generation tools like DALL-E, Midjourney, or Stable Diffusion. These prompts are designed to create consistent branding across all assets.

## Color Scheme

The branding uses the following color scheme:

- **AMD Red**: #ED1C24
- **AMD Black**: #000000
- **AMD Gray**: #58595B
- **AMD White**: #FFFFFF

Additional accent colors:

- **Teal**: #6ECACF (Coffee Supporter tier)
- **Purple**: #9C89B8 (Code Enthusiast tier)
- **Gold**: #F0A202 (ML Stack Pro tier)
- **Blue**: #58A6FF (GitHub)
- **Green**: #2EA043 (Success/Features)
