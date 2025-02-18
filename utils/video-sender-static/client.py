import cv2
import requests
import os
import argparse
import time

def send_image_to_server(image, url):
    # Encode image to JPEG format
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Lower quality (less compression)

    # Send this image as a POST request to the server
    response = requests.post(url, data=buffer.tobytes())
    print(f"Sent image to {url}; Status Code: {response.status_code}")

def main(server_url):
    # List of images in the current directory
    images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) < 3:
        print("Not enough images in the directory.")
        return
    
    try:
        # Loop through images and alternate sending them
        while True:
            for image_name in images[:3]:  # Take the first 3 images
                image = cv2.imread(image_name)
                if image is not None:
                    send_image_to_server(image, server_url)
                else:
                    print(f"Failed to load image {image_name}")

                # Wait for 1 second before sending the next image
                time.sleep(1)

    except KeyboardInterrupt:
        print("Process interrupted.")
    finally:
        print("Done sending images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send local images to server.')
    parser.add_argument('--server_url', type=str, help='URL of the server to send images to.')
    args = parser.parse_args()

    # Check if a server URL was provided as an argument, otherwise look for an environment variable
    server_url = args.server_url if args.server_url else os.getenv('SERVER_URL')

    if not server_url:
        raise ValueError("No server URL provided. Set the SERVER_URL environment variable or use the --server_url argument.")

    main(server_url)

