import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace

# Inicializa os módulos do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

model_path = 'pose_landmarker_heavy.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Cria uma instancia do pose landmarker no modo de video:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=open(model_path, "rb").read()),
    running_mode=VisionRunningMode.VIDEO)

# Abre um vídeo já gravado
cap = cv2.VideoCapture("video.mp4")

# Contadores para o relatório
frame_count = 0
emotion_count = {}
arm_up = False
arm_movements_count = 0
anomaly_count = 0  # Contador de anomalias detectadas

#Inicializa os detectores
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

  while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1  # Contabiliza o número de frames analisados
        
    # Converte a imagem para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
    #   Processa o frame para detectar faces e gestos
        results_face = face_detection.process(frame_rgb)
        results_pose = pose.process(frame_rgb)
        
    #  Se houver detecções de face, desenha os pontos e analisa expressões
        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing.draw_detection(frame, detection)
                
                # Obtém as coordenadas do retângulo da face
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                face_roi = frame[y:y+h, x:x+w]
                
                # Analisa a expressão facial com DeepFace 
                if face_roi.size > 0:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    strongest_emotion = analysis[0]['dominant_emotion']
                    
                    #Verifica se a confiança da emoção é menor do que 60% se for registra como anomalia se não salva a emoção no dicionário
                    if(analysis[0]['emotion'][strongest_emotion] < 60):
                        anomaly_count += 1
                    else:
                        cv2.putText(frame, strongest_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Registra a emoção detectada
                        if strongest_emotion in emotion_count:
                            emotion_count[strongest_emotion] += 1
                        else:
                            emotion_count[strongest_emotion] = 1     
                        
    # Função para verificar se o braço está levantado
        def is_arm_up(landmarks):
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            left_arm_up = left_elbow.y < left_eye.y
            right_arm_up = right_elbow.y < right_eye.y
            
            return left_arm_up or right_arm_up
        
        # Se houver detecções do corpo humano, desenha os pontos
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                   
            #Verifica se o braço está levantado
            if is_arm_up(results_pose.pose_landmarks.landmark):
              if not arm_up:
                arm_up = True
                arm_movements_count += 1
            else:
             arm_up = False
             
            # Exibir a contagem de movimentos dos braços no frame
            cv2.putText(frame, f'Movimentos dos bracos: {arm_movements_count}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                         
        # Exibe o frame com as detecções e emoções
        cv2.imshow("Tech Challenge Fase 4", frame)
        
        # Pressione 'q' para sair
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()

# Gera um relatório em TXT
report_text = (
    f"Total Frames Analisados: {frame_count}\n"
    f"Número de Anomalias: {anomaly_count}\n"
    f"Movimento dos Braços: {arm_movements_count}\n"
    f"Emoções Detectadas:\n"
)

for emotion, count in emotion_count.items():
    report_text += f"  {emotion}: {count}\n"   

# Salva o relatório em um arquivo TXT
with open("relatorio_analise.txt", "w") as txt_file:
    txt_file.write(report_text)

print("Relatório gerado: relatorio_analise.txt")
