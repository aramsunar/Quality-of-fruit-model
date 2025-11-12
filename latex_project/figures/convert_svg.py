"""
Simple SVG to PNG converter
Requires: pip install pillow cairosvg
"""

try:
    # Try cairosvg first (best quality)
    from cairosvg import svg2png
    svg2png(url='model_architecture.svg', write_to='model_architecture.png', output_width=2400)
    print("Converted using cairosvg - high quality PNG created")
except ImportError:
    print("cairosvg not available")
    try:
        # Try svglib + reportlab
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg('model_architecture.svg')
        renderPM.drawToFile(drawing, 'model_architecture.png', fmt='PNG', dpi=300)
        print("Converted using svglib - PNG created")
    except ImportError:
        print("\nNo conversion library available. Please install one of:")
        print("  pip install cairosvg")
        print("  pip install svglib reportlab")
        print("\nOr manually convert the SVG to PNG using an image editor.")
