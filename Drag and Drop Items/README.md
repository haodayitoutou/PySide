
# Drag and Drop Items

A simple dialog with 2 widgets in it.

Top is a GraphicsView. It has three columns of items.

Left column is a subclass of QGraphicsTextItem. It accepts a string as its text. You can drag this event and pass its text to other items. This is achieved by mousePress/MoveEvent.

Middle column is a subclass of QGraphicsPixmapItem. It can load and display an image file.

Right column is a subclass of QGraphicsItem. It can accept other item's drop and display the text it recieves. It also has an attribute to decides its background color.

All items will resize along with the GraphicsView.

Below the view is a pushbutton. Click it and I will check if the text matches the image and change right column's status accordingly. If it matches, it will be green. Otherwise it will be red.

![screenshot](https://cloud.githubusercontent.com/assets/23444620/26026669/cd6007b6-37bc-11e7-89a2-d670625428bf.PNG)
