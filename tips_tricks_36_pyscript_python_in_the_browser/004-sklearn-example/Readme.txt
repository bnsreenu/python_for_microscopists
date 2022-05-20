If we want to load an external python file as a module into the html file, the best way would be to start a local webserver. 

If the html file is located at /my_dir/my_file.html
if you just open the my_file.html in a browser, it will not recognize other files referenced..
You need to start a local webserver.

Open comand prompt (or conda prompt) - wherever you can just type python to execute python commands.
Now type: python -m http.server (or python3 - depending on how you access your local python)

This should start a webserver. Now just go to the browser and type: http://localhost:8000/
You should see a lit of files in that local directory. Now, click the html file. 