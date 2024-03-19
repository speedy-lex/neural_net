import PIL.Image

PIL.Image.open("image.png").convert("RGB").save("image.ppm")
txt=''
with open("image.ppm", "rb") as read:
    read.seek(13)
    for char in read.read()[::3]:
        txt+=str(char/255)+", "
with open("src\\image.rs", "w") as write:
    write.write("pub fn get() -> Vec<f64> {\n\tvec!["+txt[:-2]+"]\n}")