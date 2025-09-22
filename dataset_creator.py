import cv2
from utils import save_to_csv

class DatasetCreator:
    def __init__(self):
        self.dataset_mode = False
        self.manual_mode = False
        self.current_label = ""
        self.collecting = False
        self.frame_counter = 0

    def display_instructions(self, image):
        """Display current mode and instructions on the image"""
        height, width = image.shape[:2]

        cv2.rectangle(image, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (width - 10, 120), (255, 255, 255), 2)

        mode_text = "DATASET MODE: ON" if self.dataset_mode else "DATASET MODE: OFF"
        mode_color = (0, 255, 0) if self.dataset_mode else (0, 0, 255)
        cv2.putText(image, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        if self.dataset_mode:
            if self.manual_mode:
                if self.current_label:
                    status_text = f"MANUAL: {self.current_label} - Press SPACE to capture"
                    status_color = (0, 255, 0)
                else:
                    status_text = "Enter label first"
                    status_color = (255, 255, 255)
            else:
                if self.collecting:
                    status_text = f"COLLECTING: {self.current_label}"
                    status_color = (0, 255, 255)
                else:
                    status_text = "READY - Enter label and press SPACE to start"
                    status_color = (255, 255, 255)

            cv2.putText(image, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            cv2.putText(image, "Instructions:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image, "SPACE: Start/Stop | 'M': Manual | 'L': Change Label | ESC: Exit", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(image, "Press 'D' to enter Dataset Creation Mode", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "ESC: Exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def get_label_input(self):
        """Get label input from user via console with confirmation"""
        if not self.collecting and not self.manual_mode:
            print("\\n" + "=" * 10)
            print("DATASET CREATION MODE")
            print("=" * 10)

            while True:
                label = input("Enter label for data collection (or 'quit' to exit dataset mode): ").strip()

                if label.lower() == "quit":
                    self.dataset_mode = False
                    return False
                elif label:
                    print(f"\\nYou entered: '{label}'")
                    confirm = input("Is this correct? (y/n/edit): ").strip().lower()

                    if confirm in ["y", "yes"]:
                        self.current_label = label
                        print(f"✓ Label confirmed: '{self.current_label}'")
                        print("Press SPACE in the video window to start collecting data...")
                        return True
                    elif confirm in ["edit", "e"]:
                        new_label = input(f"Edit label (current: '{label}'): ").strip()
                        if new_label:
                            label = new_label
                            print(f"\\nUpdated label: '{label}'")
                            confirm2 = input("Is this correct? (y/n): ").strip().lower()
                            if confirm2 in ["y", "yes"]:
                                self.current_label = label
                                print(f"✓ Label confirmed: '{self.current_label}'")
                                print("Press SPACE in the video window to start collecting data...")
                                return True
                        else:
                            print("Empty label. Please try again.")
                    elif confirm in ["n", "no"]:
                        print("Please enter the label again.")
                    else:
                        print("Please enter 'y' for yes, 'n' for no, or 'edit' to modify the label.")
                else:
                    print("Invalid label. Please enter a valid label.")
        elif self.manual_mode:
            label = input("Enter label for manual capture: ").strip()
            if label:
                self.current_label = label
                print(f"✓ Manual mode label: '{self.current_label}'")
                return True
        return True

    def toggle_dataset_mode(self):
        """Toggle dataset creation mode"""
        self.dataset_mode = not self.dataset_mode
        self.collecting = False
        self.current_label = ""
        self.manual_mode = False
        self.frame_counter = 0
        
        mode_status = "ENTERED" if self.dataset_mode else "EXITED"
        print(f"\\n{'=' * 10}\\n{mode_status} DATASET CREATION MODE\\n{'=' * 10}")

    def toggle_manual_mode(self):
        """Toggle manual capture mode"""
        if not self.dataset_mode:
            return
            
        self.manual_mode = not self.manual_mode
        self.collecting = False
        self.current_label = ""
        
        if self.manual_mode:
            print("Switched to MANUAL capture mode")
            self.get_label_input()
        else:
            print("Switched to AUTOMATIC capture mode")

    def handle_space_key(self, results):
        """Handle space key press for data collection"""
        if not self.dataset_mode:
            return False
            
        if self.manual_mode:
            if not self.current_label:
                self.get_label_input()
            elif results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.frame_counter += 1
                    save_to_csv(hand_landmarks, self.current_label, "hand_landmarks_dataset", "manual")
                    print(f"Captured sample {self.frame_counter} for '{self.current_label}'")
                    break
            else:
                print("No hand detected - cannot capture sample")
        else:
            if not self.collecting:
                if self.get_label_input() and self.current_label:
                    self.collecting = True
                    self.frame_counter = 0
                    print(f"\\nStarted collecting data for label: '{self.current_label}'")
                    print("Press SPACE again to stop collecting...")
            else:
                self.collecting = False
                print(f"\\nStopped collecting data for label: '{self.current_label}'")
                print(f"Total frames collected: {self.frame_counter}")
                self.current_label = ""
                self.frame_counter = 0
                
                if input("Continue with another label? (y/n): ").strip().lower() != "y":
                    self.dataset_mode = False
                    print("Exited dataset creation mode.")
        return True

    def collect_frame_data(self, hand_landmarks, frame_counter):
        """Collect frame data for automatic mode"""
        if (self.collecting and self.current_label and 
            not self.manual_mode and frame_counter % 1 == 0):
            self.frame_counter += 1
            print(f"Saving frame {self.frame_counter} with label: {self.current_label}")
            save_to_csv(hand_landmarks, self.current_label, "hand_landmarks_dataset", "auto")
            return True
        return False