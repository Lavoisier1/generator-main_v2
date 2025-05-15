from io import BytesIO
from collections import defaultdict

from rdkit.Chem.Draw import rdMolDraw2D
import cairocffi as cairo
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def plot_graph_with_highlighted_contact(mol, unique_pair, save_path=None):

    colors = [
            (0.0, 0.0, 1.0, 0.1), # first contact is blue
            (1.0, 0.0, 0.0, 0.2)  # second contact is red
    ]
    athighlights = defaultdict(list)
    arads = {}
    for i in range(mol.GetNumAtoms()):
        if i in unique_pair and i == unique_pair[0]:
            athighlights[i].append(colors[0])
            arads[i] = 0.3
        elif i in unique_pair and i == unique_pair[1]:
            athighlights[i].append(colors[1])
            arads[i] = 0.3
   
    d2d = rdMolDraw2D.MolDraw2DSVG(600, 400)
    d2d.DrawMoleculeWithHighlights(mol, "", dict(athighlights), {}, arads, {})
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    if save_path:
        with open(save_path, 'w') as file:
            file.write(svg)

    return svg

def combine_svgs(svgs, output_filename):
    # Convert SVGs to cairo images
    images = [cairosvg.svg2png(bytestring=svg.encode('utf-8')) for svg in svgs]
    surfaces = [cairo.ImageSurface.create_from_png(BytesIO(img)) for img in images]

    svgs_per_row = 8
    total_width = sum(surface.get_width() for surface in surfaces[:svgs_per_row])
    total_height = sum(surface.get_height() for surface in surfaces[::svgs_per_row])
    
    # Create the output surface
    output_surface = cairo.SVGSurface(output_filename, total_width, total_height)
    ctx = cairo.Context(output_surface)

    y_offset = 0
    for idx, surface in enumerate(surfaces):
        if idx != 0 and idx % svgs_per_row == 0:
            y_offset += surfaces[0].get_height()
        
        x_offset = (idx % svgs_per_row) * surface.get_width()
        
        ctx.set_source_surface(surface, x_offset, y_offset)
        ctx.paint()

    output_surface.finish()

def plot_mol_contacts(smiles_string, sorted_contacts):
    mol = Chem.MolFromSmiles(smiles_string)
    Chem.Kekulize(mol)
    svg_data = [plot_graph_with_highlighted_contact(mol, unique_contact) for unique_contact in sorted_contacts]
    combine_svgs(svg_data, f"{smiles_string}.svg")

def combine_svg_png(svg_path, png_path, lccc, rccc, output_path):
    # Convert SVG to PNG
    svg_data = open(svg_path, 'rb').read()  # Read the SVG data
    png_from_svg = cairosvg.svg2png(bytestring=svg_data)

    # Read the original PNG
    original_png = mpimg.imread(png_path)

    # Read the PNG converted from SVG
    converted_png = mpimg.imread(BytesIO(png_from_svg))

    # Plotting both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(converted_png)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title(f"Molecule with contacts at {lccc} {rccc}")

    axes[1].imshow(original_png)
    axes[1].axis('off')
#    axes[1].set_title("Transmission")

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)

def combine_svg_png_return_image(svg_path, png_image, lccc, rccc):
    # Convert SVG to PNG
    svg_data = open(svg_path, 'rb').read()
    png_from_svg = cairosvg.svg2png(bytestring=svg_data)

    # Read the PNG converted from SVG
    converted_png = Image.open(BytesIO(png_from_svg))

    # Convert PIL Image to matplotlib image for displaying
    converted_png_for_plot = plt.imread(BytesIO(png_from_svg))

    # Plotting both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(converted_png_for_plot)
    axes[0].axis('off')
    axes[0].set_title(f"Molecule with contacts at {lccc} {rccc}")

    axes[1].imshow(png_image)
    axes[1].axis('off')

    plt.tight_layout()

    # Instead of saving to BytesIO, directly save to a PNG buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Create a new Image object that is independent of BytesIO
    combined_image = Image.open(buf).copy()

    plt.close(fig)
    buf.close()

    return combined_image

# Function to combine the images into a single image
def combine_all_images(images, spacing=1):
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    total_height = sum(heights) + (len(images) - 1) * spacing
    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + spacing

    return new_im

def plot_transmission(rdkit_mol, energy, transmission, molecule, lccc, rccc):

    plot_graph_with_highlighted_contact(rdkit_mol, (lccc, rccc), molecule)
    plt.figure(figsize=(10, 6))
    plt.plot(energy, transmission, label='Transmission Coefficient')
    plt.title('Transmission vs Energy')
    plt.xlabel('Energy (arb. units)')
    plt.ylabel('Transmission')
    plt.grid(True)
    plt.legend()
    plt.show()
    png_file = f"{molecule}_{lccc}_{rccc}.png"
    plt.savefig(png_file)

    # Save the matplotlib figure to a BytesIO object instead of a file
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img1 = Image.open(buf)
    combined_img = combine_svg_png_return_image(molecule, img1, lccc, rccc)
    buf.close()
    plt.close()

    return combined_img

