# Quick Launch Manual

**MARTApp** is available in two flavors:

- X11-based, which is a basic X11 forwarding and the user natively uses the application in its system.
- noVNC-based, which allows the user to use the application in its browser.

For both, Docker is required. It can be downloaded from [here](https://www.docker.com/products/docker-desktop/). Once installed, we can proceed pulling MARTApp using your
system command line (distributed through [GitHub registry](https://github.com/orgs/ALBA-Synchrotron/packages/container/package/martapp)):

```bash
docker pull ghcr.io/alba-synchrotron/martapp:latest # or martapp:latest_novnc for the noVNC version
```

Then, to launch the application, the process is different depending on the chosen falvor.

## X11-based

To use X11, depending on the system some different previous steps are required:

<details>
<summary>Linux</summary>
First, enable X11 forwarding:

```bash
xhost +local:docker
```

Finally, start the container:

```bash
docker run -e DISPLAY=$DISPLAY \
           -v "/tmp/.X11-unix/:/tmp/.X11-unix/" \
           -v "SOURCE:DESTINATION" \
           ghcr.io/alba-synchrotron/martapp:latest
```
- `-e DISPLAY=$DISPLAY` allows us to access the display for the GUI.
- `-v "/tmp/.X11-unix/:/tmp/.X11-unix/"` allows us to forward X11.
- `-v "SOURCE:DESTINATION"` allows us to access SOURCE path (in our machine) as the path indicated in DESTINATION.
</details>

<details>
<summary>Windows</summary>

Firstly, we need to install **XLaunch**, to do so we need to install **VcXsrv Windows X Server** that can be downloaded from [here](https://sourceforge.net/projects/vcxsrv/) (using default settings/installation). Then, open and setup **XLaunch**:

1. Select "Multiple windows".
2. Choose "Start no client".
3. Ensure "Clipboard" is checked to allow copying between Windows and the application.
4. Check "Native OpenGL".
5. Finish and keep it runing.

Then, we need to know the IP of our computer, we can use ipconfig:
```bash
>> ipconfig
...
Adaptador de Ethernet Ethernet:
   Dirección IPv4. . . . . . . . . . . . . . : XXX.XXX.XXX.XXX # This is the IP of our computer
...
```

Finally, we can launch the application:
```bash
docker run -e DISPLAY=COMPUTER_IP \
           -v "/tmp/.X11-unix/:/tmp/.X11-unix/" \
           -v "SOURCE:DESTINATION" \
           ghcr.io/alba-synchrotron/martapp:latest
```

- `-e DISPLAY=COMPUTER_IP` allows us to access the display for the GUI. `COMPUTER_IP` must be our IP retrieved using ipconfig. For some cases instead of using the IP we can set `DISPLAY=host.docker.internal:0.0`.
- `-v "/tmp/.X11-unix/:/tmp/.X11-unix/"` allows us to forward X11.
- `-v "SOURCE:DESTINATION"` allows us to access `SOURCE` path (in our machine) as the path indicated in `DESTINATION`.

</details>

<details>
<summary>macOS</summary>

First, we need to install **XQuartz** that can be download from here or using the command line:
```bash
brew install --cask xquartz
```

After restarting MacOS we should do the following using the command line:
```bash
# Open XQuartz
open -a XQuartz

# Enable "Allow connections from network clients" option in Preferences>Security

# Add localhost as an allowed source in order to share the screen
xhost + 127.0.0.1
```

Finally, we can launch the application:
```bash
docker run -e DISPLAY=host.docker.internal:0 \
           -v "/tmp/.X11-unix:/tmp/.X11-unix" \
           -v "SOURCE:DESTINATION" \
           ghcr.io/alba-synchrotron/martapp:latest
```
- `-e DISPLAY=host.docker.internal:0` allows us to access the display for the GUI.
- `-v "/tmp/.X11-unix:/tmp/.X11-unix"` allows us to forward X11.
- `-v "SOURCE:DESTINATION"` allows us to access `SOURCE` path (in our machine) as the path indicated in `DESTINATION`.

</details>

## noVNC-based


<details>
<summary>noVNC-based</summary>

We only need to launch the application:

```bash
docker run -p 5900:5900 -p 6080:6080 \
           -v "SOURCE:DESTINATION" \
           ghcr.io/alba-synchrotron/martapp:latest_novnc

# The application will be automatically open in your browser. If not, enter in http://localhost:6080/ .
```
- `-p 5900:5900 -p 6080:6080` allows port mapping between the docker container and the system.
- `-v "SOURCE:DESTINATION"` allows us to access `SOURCE` path (in our machine) as the path indicated in `DESTINATION`.

If the ports are already in use by other applications, change them (e.g, increasing by 1 the numbers), and by they are used by docker (e.g. because the application has been incorrectly closed) the following can be done:
```bash
# Look for the docker container ID
docker ps

# Kill the container
docker <CONTAINER_ID>

# Run again the application
```

</details>