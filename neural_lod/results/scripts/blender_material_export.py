import bpy

folder = "output_texture_folder"

width = 4
height = 4

specular_name = "Specular IOR Level" if bpy.app.version > (4, 0, 0) else "Specular"
transmission_name = "Transmission Weight" if bpy.app.version > (4, 0, 0) else "Transmission"

def set_pixel(img,x,y,color):
    offs = (x + int(y*width)) * 4
    for i in range(4):
        img.pixels[offs+i] = color[i]


for mat in bpy.data.materials:
    print(mat.name)
    base_color_image = bpy.data.images.new(mat.name, alpha=False, width=width, height=height)
    base_color_image.filepath_raw = folder + mat.name + "_BaseColor.png"
    base_color_image.file_format = 'PNG'
    base_color = (0,0,0,1)
    specular = 0
    roughness = 0
    metallic = 0
    specular_transmission = 0
    ior = 0
    
    found_texture = False
    
    if mat.use_nodes:
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_DIFFUSE':
                base_color  = n.inputs[0].default_value[:]
            if n.type == "BSDF_PRINCIPLED":
                base_color = n.inputs["Base Color"].default_value[:]
                specular = n.inputs[specular_name].default_value
                roughness = n.inputs["Roughness"].default_value
                metallic = n.inputs["Metallic"].default_value
                specular_transmission = n.inputs[transmission_name].default_value
                ior = n.inputs["IOR"].default_value
    else:
        print("Use nodes and a principled bsdf material")
                
    if not found_texture:
        for x in range(4):
            for y in range(4):
                set_pixel(base_color_image,x,y,base_color)
    base_color_image.save()
    
    specular_image = bpy.data.images.new(mat.name, alpha=False, width=width, height=height)
    specular_image.filepath_raw = folder + mat.name + "_Specular.png"
    specular_image.file_format = 'PNG'
    for x in range(4):
        for y in range(4):
            set_pixel(specular_image,x,y,(specular,roughness,metallic,1))
    specular_image.save()
    
    if specular_transmission > 0:
        with open(folder + mat.name + "_SpecularTransmission.txt",'w') as file_object:
            file_object.write(str(specular_transmission))
            
        with open(folder + mat.name + "_IorEta.txt",'w') as file_object:
            file_object.write(str(ior))
    
    
