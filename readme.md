Julia Set Sequence Generator
============================
> Just because i can™

__Requirements:__

* PyCUDA
* NumPy
* Matplotlib

__Run:__

    python julia.py --dir dir_name

__Optional Arguments:__

    --dpi       Image density
    --width     Image width
    --height    Image height
    --frames    Number of frames
    --stepr     Real Step
    --stepi     Imaginary Step
    --real      Starting point
    --imag      Starting point
    --norm      Color normal
    --dir       Output directory
    --prefix    Output prefix
    --cmap      Color map
    --xfrom     X start
    --xto       X end
    --yfrom     Y start
    --yto       Y end
    --plotc     Plot c values


__Results:__

Some [Videos](https://www.youtube.com/watch?v=D4abkM5rBDI&list=UUTyhIPFP9Tf9LtRf8Ah9OLA) generated from program output using ffmpeg

     ffmpeg -r 30 -i out/julia_%04d.png -async 44100 -qscale 1 -vcodec mpeg4 -y movie.mp4