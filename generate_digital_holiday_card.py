"""
Create your own digital card for the Holidays with a Christmas tree 
and your custom message. 

    - Sreenivas Bhattiprolu

"""
from PIL import Image, ImageDraw, ImageFont

def generate_holiday_greeting(message, font_size=15, image_size=(300, 400), background_color=(0, 0, 0), text_color=(255, 255, 0)):
    # Create a blank image
    image = Image.new('RGB', image_size, background_color)

    # Get drawing context
    draw = ImageDraw.Draw(image)

 # Generate a Christmas tree
    tree_height = 15
    for i in range(tree_height):
        stars = "*" * (2 * i + 1)
        spaces = " " * (tree_height - i)
        row = spaces + stars
        draw.text((60, 20 + i * 20), row, font=None, fill=(34, 139, 34))  # Forest Green

    # Set the trunk position
    trunk_height = 3

    # Add trunk to the Christmas tree
    for i in range(trunk_height):
        draw.text((145, 310 + i * 10), "|||", font=None, fill=(139, 69, 19))  # Saddle Brown


    # Load a larger font with the specified size
    #I am using Lobster font from Google that I downloaded
    #You can use default like arial, if you do not have Lobster or other fancy fonts
    font = ImageFont.truetype("Lobster-Regular.ttf", font_size) # arial.ttf

    # Adjust the text position to center it
    
    text_width  = draw.textlength(message, font)
    text_height = -300
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2

    # Draw text on the image
    draw.text((x, y), message, font=font, fill=text_color)

    # Save and display the image
    image.save('holiday_greeting.png')
    image.show()

if __name__ == "__main__":
    holiday_message = "Happy Holidays from @DigitalSreeni!"

    generate_holiday_greeting(holiday_message)
