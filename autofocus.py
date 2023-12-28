import cv2
import numpy as np

def calculate_energy(img):
    # Compute 2D spectrum
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Log-scaled for better visualization

    # Calculate energy (average of squared magnitude)
    energy = np.mean(np.square(np.abs(fshift)))
    
    return energy, magnitude_spectrum

def main():
    # Open a connection to the webcam (you may need to change the index)
    cap = cv2.VideoCapture(1)

    # Set an initial focus position
    initial_focus_position = 0

    # Number of focus positions to try
    num_positions = 10

    # Adjust focus and measure energy for each position
    for i in range(num_positions):
        # Adjust focus (you may need to use a library specific to your camera)
        # For simplicity, we just change the initial_focus_position
        current_focus_position = initial_focus_position + i

        # Capture an image
        ret, frame = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate energy of the spectrum
        energy, spectrum = calculate_energy(gray)

        # Display the original image and its spectrum
        cv2.imshow('Original Image', gray)
        cv2.imshow('Spectrum', spectrum)

        print(f"Focus Position: {current_focus_position}, Energy: {energy}")

        # Wait for a key press
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
