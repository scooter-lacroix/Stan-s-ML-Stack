#!/usr/bin/env python3
# =============================================================================
# Generate Images
# =============================================================================
# This script generates images for the ML Stack branding.
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

import os
import sys
import argparse
import json
from pathlib import Path

# Define the image prompts
IMAGE_PROMPTS = {
    "logo": {
        "file_name": "ml_stack_logo.png",
        "prompt": """
A professional, modern logo for "Stan's ML Stack" featuring AMD GPU technology. 
The logo should incorporate the AMD red color (#ED1C24) and show a stylized stack of GPUs with neural network connections. 
The design should be clean, minimalist, and suitable for both light and dark backgrounds. 
Include subtle references to machine learning and GPU computing.
Highly detailed, professional quality, vector art style.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "github_banner": {
        "file_name": "github_banner.png",
        "prompt": """
A wide banner image (1280x640px) for a GitHub repository called "Stan's ML Stack". 
The banner should feature AMD Radeon GPUs (specifically the RX 7900 XTX and RX 7800 XT) with glowing red accents. 
The background should be dark with code snippets and neural network visualizations. 
Include the text "Stan's ML Stack" in a modern tech font. 
The overall style should be professional and high-tech, suitable for a machine learning repository focused on AMD GPU optimization.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "watermark, signature, blurry, low quality, amateur"
    },
    "coffee_supporter": {
        "file_name": "coffee_supporter.png",
        "prompt": """
A professional illustration for a Patreon tier called "Coffee Supporter" ($5/month) for an AMD GPU machine learning project. 
The image should feature a stylized coffee cup with steam that forms into neural network patterns. 
Use a teal color scheme (#6ECACF) with subtle AMD red accents. 
The design should be clean, modern, and appealing to tech enthusiasts.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "code_enthusiast": {
        "file_name": "code_enthusiast.png",
        "prompt": """
A professional illustration for a Patreon tier called "Code Enthusiast" ($10/month) for an AMD GPU machine learning project. 
The image should feature stylized code snippets and GPU architecture elements. 
Use a purple color scheme (#9C89B8) with subtle AMD red accents. 
The design should convey the idea of diving deeper into machine learning code and GPU optimization.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "ml_stack_pro": {
        "file_name": "ml_stack_pro.png",
        "prompt": """
A professional illustration for a Patreon tier called "ML Stack Pro" ($25/month) for an AMD GPU machine learning project. 
The image should feature a premium, high-end visualization of AMD Radeon GPU hardware with glowing elements and advanced neural network structures. 
Use a gold color scheme (#F0A202) with AMD red accents. 
The design should convey expertise, optimization, and professional-level machine learning development.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "profile_image": {
        "file_name": "profile_image.png",
        "prompt": """
A professional profile picture for a machine learning developer specializing in AMD GPU optimization. 
The image should be a stylized avatar that incorporates elements of AMD GPU technology and neural networks. 
Use the AMD red color (#ED1C24) as an accent color with a dark background. 
The design should be simple enough to be recognizable at small sizes while conveying expertise in GPU-accelerated machine learning.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur, photograph, realistic, photo-realistic"
    },
    "documentation_header": {
        "file_name": "documentation_header.png",
        "prompt": """
A wide header image (1200x300px) for documentation pages of "Stan's ML Stack". 
The image should visualize the architecture of a machine learning stack optimized for AMD GPUs. 
Include representations of ROCm, PyTorch, ONNX, MIGraphX, and Megatron-LM in a flowing, connected design. 
Use a color scheme that incorporates AMD red (#ED1C24) with complementary blues and purples. 
The style should be technical yet approachable, with subtle grid patterns and code elements in the background.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "benchmark_visualization": {
        "file_name": "benchmark_visualization.png",
        "prompt": """
A professional data visualization image for machine learning benchmarks comparing performance between AMD Radeon RX 7900 XTX and RX 7800 XT GPUs. 
The image should feature stylized bar charts or line graphs showing performance metrics with the 7900 XTX outperforming the 7800 XT. 
Use AMD red (#ED1C24) for the 7900 XTX and a complementary color for the 7800 XT. 
Include subtle neural network patterns in the background. 
The visualization should be clean, modern, and data-focused while maintaining visual appeal.
Highly detailed, professional quality, digital art.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur"
    },
    "installation_icon": {
        "file_name": "installation_icon.png",
        "prompt": """
A simple, professional icon for an installation guide section of "Stan's ML Stack" documentation. 
The icon should represent the installation of machine learning software on AMD GPUs. 
Incorporate elements like a download symbol, AMD GPU, and a simplified ROCm logo. 
Use AMD red (#ED1C24) as the primary color with a clean, minimalist design that works well at small sizes.
Highly detailed, professional quality, vector art style.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur, photograph, realistic, photo-realistic"
    },
    "troubleshooting_icon": {
        "file_name": "troubleshooting_icon.png",
        "prompt": """
A simple, professional icon for a troubleshooting guide section of "Stan's ML Stack" documentation. 
The icon should represent fixing issues with machine learning software on AMD GPUs. 
Incorporate elements like a wrench or gear with a stylized GPU and warning symbol. 
Use AMD red (#ED1C24) as an accent color with a clean, minimalist design that works well at small sizes.
Highly detailed, professional quality, vector art style.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur, photograph, realistic, photo-realistic"
    },
    "performance_icon": {
        "file_name": "performance_icon.png",
        "prompt": """
A simple, professional icon for a performance optimization section of "Stan's ML Stack" documentation. 
The icon should represent improving machine learning performance on AMD GPUs. 
Incorporate elements like a speedometer, GPU, and upward trending graph. 
Use AMD red (#ED1C24) as an accent color with a clean, minimalist design that works well at small sizes.
Highly detailed, professional quality, vector art style.
        """,
        "negative_prompt": "text, watermark, signature, blurry, low quality, amateur, photograph, realistic, photo-realistic"
    }
}

def save_prompts_to_file():
    """Save the prompts to a JSON file."""
    output_dir = Path("images/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "prompts.json"
    
    with open(output_file, "w") as f:
        json.dump(IMAGE_PROMPTS, f, indent=4)
    
    print(f"Prompts saved to {output_file}")

def save_prompts_to_markdown():
    """Save the prompts to a markdown file."""
    output_dir = Path("images/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "prompts.md"
    
    with open(output_file, "w") as f:
        f.write("# Image Generation Prompts for Stan's ML Stack\n\n")
        
        for key, data in IMAGE_PROMPTS.items():
            f.write(f"## {key.replace('_', ' ').title()}\n\n")
            f.write(f"**File Name:** {data['file_name']}\n\n")
            f.write("### Prompt\n\n")
            f.write("```\n")
            f.write(data["prompt"].strip())
            f.write("\n```\n\n")
            f.write("### Negative Prompt\n\n")
            f.write("```\n")
            f.write(data["negative_prompt"].strip())
            f.write("\n```\n\n")
    
    print(f"Prompts saved to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate images for the ML Stack branding.")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both",
                        help="Output format for the prompts (default: both)")
    
    args = parser.parse_args()
    
    if args.format in ["json", "both"]:
        save_prompts_to_file()
    
    if args.format in ["markdown", "both"]:
        save_prompts_to_markdown()
    
    print("\nTo generate images, use these prompts with an AI image generation tool like:")
    print("- DALL-E")
    print("- Midjourney")
    print("- Stable Diffusion")
    print("\nSave the generated images to the appropriate directories in the 'images' folder.")

if __name__ == "__main__":
    main()
