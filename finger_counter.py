import cv2
import mediapipe as mp
import time

class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]

    def count_fingers(self, landmarks):
        fingers_up = []
        
        if landmarks[self.finger_tips[0]].x > landmarks[self.finger_pips[0]].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        for i in range(1, 5):
            if landmarks[self.finger_tips[i]].y < landmarks[self.finger_pips[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
                
        return sum(fingers_up)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return
            
        print("Счётчик пальцев запущен!")
        print("Покажите руку камере")
        print("Нажмите 'q' для выхода")
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: Не удалось прочитать кадр")
                break
                
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            total_fingers = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    fingers = self.count_fingers(hand_landmarks.landmark)
                    total_fingers += fingers
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(frame, f"Paltsev: {total_fingers}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "Nazhmi 'q' dlya vykhoda", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Schetchik paltsev', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        counter = FingerCounter()
        counter.run()
    except Exception as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что установлены необходимые пакеты:")
        print("pip install opencv-python mediapipe")

if __name__ == "__main__":
    main()