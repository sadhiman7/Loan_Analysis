from pdf2image import convert_from_path

pages = convert_from_path('input/input.pdf', poppler_path = r"C:\Program Files\poppler-0.68.0\bin")

for i, image in enumerate(pages):
    fname = "image" + str(i) + ".png"
    output_path = "output/" + fname
    image.save(output_path, "PNG")