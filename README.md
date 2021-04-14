steps to run the code:
1. clone the repository from Github

    git clone https://github.com/rputh055/dominant_color

2. change directory to the repository folder

    cd dominant_color

3. install the required modules.

    pip install -r requirements.txt

4. run the file.

    python3 main.py

This will create folders for RED, GREEN, BLUE and OTHER and the images with dominant color that matches with any of the class will be saved to their respective folders.

Approach:

1. To solve this problem I have used opencv to perform operations on the images.
2. Used KMeans clustering to segregate the colors into clusters.
3. Created a histogram to know which cluster has the highest percentage of color.
4. Compared the RGB values of dominant color with RGB values of red, green and blue and got their differences.
5. Set a threshold and if the difference is less than threshold then the dominant color belongs to that class.
6. Created seperate folders for each class and saved the images that belongs to that class.
