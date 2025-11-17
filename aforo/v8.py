
"""
Sistema de Control de Aforo con Calibración de Altura por Puerta
Permite calibrar la altura real ingresando la altura de una puerta o referencia
"""

import cv2
import mysql.connector
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict


# =============================================================
# === LECTOR RTSP SIN LAG =====================================
# =============================================================

class RTSPReader:
    def __init__(self, fuente):
        self.stream = cv2.VideoCapture(fuente, cv2.CAP_FFMPEG)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame_actual = None
        self.running = True

        import threading
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if ret:
                self.frame_actual = frame

    def read(self):
        return self.frame_actual

    def stop(self):
        self.running = False
        self.stream.release()


# =============================================================
# === CLASE PRINCIPAL DEL SISTEMA =============================
# =============================================================

class ControlAforoCruceLinea:
    def __init__(self, db_config):
        self.db_config = db_config
        self.modelo = None
        self.conexion_db = None
        self.cursor = None
        
        self.y_cruce = None  
        self.posicion_y_cruce = 0.5  
        self.offset_zona = 40  
        
        self.personas_trackeadas = {}
        self.siguiente_id = 0
        self.distancia_maxima_tracking = 100  
        
        self.historial_posicion = defaultdict(list)
        self.max_historial = 5  
        
        self.ids_cruzados = set() 
        
        self.total_entradas = 0
        self.total_salidas = 0
        
        # ===== CALIBRACIÓN DE ALTURA =====
        self.calibrado = False
        self.altura_real_referencia_cm = None  # Altura real de la puerta/referencia en cm
        self.altura_pixeles_referencia = None  # Altura en píxeles de la puerta/referencia
        self.factor_conversion = None  # cm por píxel
        
        # Modo de calibración
        self.modo_calibracion = False
        self.punto1_calibracion = None
        self.punto2_calibracion = None
    

    # =========================================================
    # === CALIBRACIÓN DE ALTURA ===============================
    # =========================================================

    def iniciar_calibracion(self, altura_real_cm):
        """
        Inicia el modo de calibración
        altura_real_cm: altura real en cm de la referencia (ej: puerta de 210cm)
        """
        self.altura_real_referencia_cm = altura_real_cm
        self.modo_calibracion = True
        self.punto1_calibracion = None
        self.punto2_calibracion = None
        print(f"\n{'='*60}")
        print(f"MODO CALIBRACIÓN ACTIVADO")
        print(f"{'='*60}")
        print(f"Altura de referencia: {altura_real_cm} cm")
        print(f"\nInstrucciones:")
        print(f"1. Haz clic en la PARTE SUPERIOR de la puerta/referencia")
        print(f"2. Haz clic en la PARTE INFERIOR de la puerta/referencia")
        print(f"3. Presiona 'C' para confirmar la calibración")
        print(f"4. Presiona 'R' para reiniciar la calibración")
        print(f"{'='*60}\n")

    def mouse_callback_calibracion(self, event, x, y, flags, param):
        """Callback para capturar los puntos de calibración"""
        if event == cv2.EVENT_LBUTTONDOWN and self.modo_calibracion:
            if self.punto1_calibracion is None:
                self.punto1_calibracion = (x, y)
                print(f"✓ Punto superior marcado en: ({x}, {y})")
            elif self.punto2_calibracion is None:
                self.punto2_calibracion = (x, y)
                altura_px = abs(self.punto2_calibracion[1] - self.punto1_calibracion[1])
                print(f"✓ Punto inferior marcado en: ({x}, {y})")
                print(f"✓ Altura en píxeles: {altura_px}")
                print(f"\nPresiona 'C' para CONFIRMAR o 'R' para REINICIAR")

    def confirmar_calibracion(self):
        """Confirma y guarda la calibración"""
        if self.punto1_calibracion and self.punto2_calibracion:
            self.altura_pixeles_referencia = abs(self.punto2_calibracion[1] - self.punto1_calibracion[1])
            self.factor_conversion = self.altura_real_referencia_cm / self.altura_pixeles_referencia
            self.calibrado = True
            self.modo_calibracion = False
            
            print(f"\n{'='*60}")
            print(f"✓ CALIBRACIÓN COMPLETADA")
            print(f"{'='*60}")
            print(f"Altura referencia: {self.altura_real_referencia_cm} cm")
            print(f"Altura en píxeles: {self.altura_pixeles_referencia} px")
            print(f"Factor conversión: {self.factor_conversion:.4f} cm/px")
            print(f"{'='*60}\n")
            return True
        else:
            print("✗ Error: Debes marcar ambos puntos antes de confirmar")
            return False

    def reiniciar_calibracion(self):
        """Reinicia los puntos de calibración"""
        self.punto1_calibracion = None
        self.punto2_calibracion = None
        print("\n⟲ Calibración reiniciada. Marca los puntos nuevamente.")

    def estimar_altura_real(self, altura_px):
        """Estima altura real en cm basándose en la calibración"""
        if not self.calibrado or self.factor_conversion is None:
            return None
        
        altura_estimada = altura_px * self.factor_conversion
        return round(altura_estimada, 1)
    

    # =========================================================
    # === MÉTODOS DE BASE DE DATOS ============================
    # =========================================================

    def conectar_db(self):
        try:
            self.conexion_db = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                autocommit=False,
                pool_reset_session=True,
                connection_timeout=10
            )
            self.cursor = self.conexion_db.cursor()
            print("✓ Conexión a MySQL establecida correctamente")
            return True
        except mysql.connector.Error as e:
            print(f"✗ Error al conectar a MySQL: {e}")
            return False
    
    def verificar_reconectar_db(self):
        """Verifica conexión y reconecta si es necesario"""
        try:
            if self.conexion_db is None or not self.conexion_db.is_connected():
                print("⚠ Reconectando a la base de datos...")
                return self.conectar_db()
            return True
        except:
            print("⚠ Reconectando a la base de datos...")
            return self.conectar_db()
    

    def crear_tablas(self):
        queries = [
            """
            CREATE TABLE IF NOT EXISTS movimientos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fecha_hora DATETIME NOT NULL,
                tipo_movimiento ENUM('entrada', 'salida') NOT NULL,
                numero_personas INT NOT NULL DEFAULT 1,
                altura_pixeles FLOAT,
                altura_estimada_cm FLOAT,
                clasificacion VARCHAR(50),
                camara VARCHAR(100) NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS aforo_actual (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fecha_hora_actualizacion DATETIME NOT NULL,
                personas_dentro INT NOT NULL DEFAULT 0,
                total_entradas INT NOT NULL DEFAULT 0,
                total_salidas INT NOT NULL DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS historial_aforo (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fecha_hora DATETIME NOT NULL,
                personas_dentro INT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS calibracion_camara (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fecha_hora DATETIME NOT NULL,
                camara VARCHAR(100) NOT NULL,
                altura_referencia_cm FLOAT NOT NULL,
                altura_referencia_px FLOAT NOT NULL,
                factor_conversion FLOAT NOT NULL
            );
            """
        ]
        
        try:
            for query in queries:
                self.cursor.execute(query)
            self.conexion_db.commit()
            print("✓ Tablas verificadas/creadas correctamente")
            
            self.cursor.execute("SELECT COUNT(*) FROM aforo_actual")
            if self.cursor.fetchone()[0] == 0:
                self.cursor.execute("""
                    INSERT INTO aforo_actual (fecha_hora_actualizacion, personas_dentro, total_entradas, total_salidas)
                    VALUES (NOW(), 0, 0, 0)
                """)
                self.conexion_db.commit()
                print("✓ Aforo inicial configurado en 0")
                
        except mysql.connector.Error as e:
            print(f"✗ Error al crear tablas: {e}")
    
    def guardar_calibracion_db(self, nombre_camara):
        """Guarda los datos de calibración en la base de datos"""
        if not self.calibrado:
            return False
        
        try:
            fecha = datetime.now()
            self.cursor.execute(
                """INSERT INTO calibracion_camara 
                   (fecha_hora, camara, altura_referencia_cm, altura_referencia_px, factor_conversion) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (fecha, nombre_camara, self.altura_real_referencia_cm, 
                 self.altura_pixeles_referencia, self.factor_conversion)
            )
            self.conexion_db.commit()
            print(f"✓ Calibración guardada en base de datos")
            return True
        except mysql.connector.Error as e:
            print(f"✗ Error al guardar calibración: {e}")
            return False
    

    # =========================================================
    # === CLASIFICACIÓN POR ALTURA ============================
    # =========================================================

    def calcular_altura_persona(self, bbox):
        """Calcula la altura en píxeles del bounding box"""
        x1, y1, x2, y2 = bbox
        altura_px = y2 - y1
        return altura_px
    
    def clasificar_persona(self, altura_px, ancho_px):
        """
        Clasifica persona por altura calibrada
        Si no hay calibración, retorna clasificación genérica
        """
        altura_cm = self.estimar_altura_real(altura_px)
        
        if altura_cm is None:
            # Sin calibración
            return "Sin calibrar", None, (128, 128, 128)
        
        proporcion = altura_px / ancho_px if ancho_px > 0 else 0
        
        # Clasificación por altura real
        if altura_cm < 110:
            categoria = "Niño/a"
            color = (255, 200, 0)  # Azul claro
        elif altura_cm < 150:
            categoria = "Adolescente"
            color = (0, 200, 255)  # Naranja
        else:
            # Distinguir adultos por proporción (estimación básica)
            if proporcion > 2.8:
                categoria = "Mujer"
                color = (255, 0, 255)  # Magenta
            else:
                categoria = "Hombre"
                color = (255, 100, 0)  # Azul
        
        return categoria, altura_cm, color
    

    # =========================================================
    # === HERRAMIENTAS ========================================
    # =========================================================

    def calcular_distancia_euclidiana(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def obtener_centro_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2)/2), int((y1 + y2)/2))
    
    def obtener_pies_persona(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2)/2), int(y2))
    

    # =========================================================
    # === TRACKING ============================================
    # =========================================================

    def trackear_personas(self, detecciones, altura):
        personas_actual = {}
        ids_usados = set()
        
        for bbox in detecciones:
            centro = self.obtener_centro_bbox(bbox)
            pies = self.obtener_pies_persona(bbox)
            
            altura_px = self.calcular_altura_persona(bbox)
            ancho_px = bbox[2] - bbox[0]
            clasificacion, altura_cm, color = self.clasificar_persona(altura_px, ancho_px)
            
            mejor_id = None
            mejor_dist = self.distancia_maxima_tracking
            
            for id_persona, info in self.personas_trackeadas.items():
                if id_persona in ids_usados:
                    continue
                
                dist = self.calcular_distancia_euclidiana(centro, info['centro'])
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id = id_persona
            
            if mejor_id is not None:
                personas_actual[mejor_id] = {
                    "bbox": bbox,
                    "centro": centro,
                    "pies": pies,
                    "altura_px": altura_px,
                    "altura_cm": altura_cm,
                    "clasificacion": clasificacion,
                    "color": color
                }
                ids_usados.add(mejor_id)
            else:
                nuevo_id = self.siguiente_id
                self.siguiente_id += 1
                personas_actual[nuevo_id] = {
                    "bbox": bbox,
                    "centro": centro,
                    "pies": pies,
                    "altura_px": altura_px,
                    "altura_cm": altura_cm,
                    "clasificacion": clasificacion,
                    "color": color
                }
        
        ids_actuales = set(personas_actual.keys())
        ids_fuera = set(self.personas_trackeadas.keys()) - ids_actuales
        
        for id_del in ids_fuera:
            if id_del in self.historial_posicion:
                del self.historial_posicion[id_del]
            if id_del in self.ids_cruzados:
                self.ids_cruzados.remove(id_del)
        
        self.personas_trackeadas = personas_actual
        return personas_actual
    

    # =========================================================
    # === CRUCE DE LÍNEA ======================================
    # =========================================================

    def detectar_cruce_linea(self, id_persona, y_pies):
        self.historial_posicion[id_persona].append(y_pies)
        if len(self.historial_posicion[id_persona]) > self.max_historial:
            self.historial_posicion[id_persona].pop(0)

        if len(self.historial_posicion[id_persona]) < 2:
            return None

        y_cruce = self.y_cruce
        zona_actual = "Arriba" if y_pies < y_cruce else "Abajo"
        y_inicial = np.mean(self.historial_posicion[id_persona][:2])
        zona_anterior = "Arriba" if y_inicial < y_cruce else "Abajo"
        movimiento = None

        if zona_actual != zona_anterior:
            if id_persona not in self.ids_cruzados:
                if zona_anterior == "Arriba" and zona_actual == "Abajo":
                    movimiento = "entrada"
                elif zona_anterior == "Abajo" and zona_actual == "Arriba":
                    movimiento = "salida"

                if movimiento:
                    self.ids_cruzados.add(id_persona)
                    return movimiento
        
        return None
    

    # =========================================================
    # === DETECCIÓN + TRACKING ================================
    # =========================================================

    def detectar_y_trackear(self, frame):
        altura = frame.shape[0]
        ancho = frame.shape[1]

        if self.y_cruce is None:
            self.y_cruce = int(altura * self.posicion_y_cruce)

        resultados = self.modelo(frame, classes=[0], verbose=False)

        detecciones = []
        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if conf > 0.5:
                    detecciones.append((int(x1), int(y1), int(x2), int(y2)))

        personas = self.trackear_personas(detecciones, altura)
        frame_out = frame.copy()
        movimientos = []

        y_cruce = self.y_cruce

        # ===== MODO CALIBRACIÓN =====
        if self.modo_calibracion:
            # Dibujar puntos de calibración
            if self.punto1_calibracion:
                cv2.circle(frame_out, self.punto1_calibracion, 10, (0, 255, 0), -1)
                cv2.putText(frame_out, "Punto Superior", 
                           (self.punto1_calibracion[0] + 15, self.punto1_calibracion[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.punto2_calibracion:
                cv2.circle(frame_out, self.punto2_calibracion, 10, (0, 255, 0), -1)
                cv2.putText(frame_out, "Punto Inferior", 
                           (self.punto2_calibracion[0] + 15, self.punto2_calibracion[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.punto1_calibracion and self.punto2_calibracion:
                # Dibujar línea entre puntos
                cv2.line(frame_out, self.punto1_calibracion, self.punto2_calibracion, 
                        (0, 255, 0), 3)
                altura_px = abs(self.punto2_calibracion[1] - self.punto1_calibracion[1])
                punto_medio = ((self.punto1_calibracion[0] + self.punto2_calibracion[0]) // 2,
                              (self.punto1_calibracion[1] + self.punto2_calibracion[1]) // 2)
                cv2.putText(frame_out, f"{altura_px}px = {self.altura_real_referencia_cm}cm", 
                           punto_medio, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Instrucciones en pantalla
            cv2.rectangle(frame_out, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.putText(frame_out, "MODO CALIBRACION", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_out, f"Altura referencia: {self.altura_real_referencia_cm}cm", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_out, "Click: marcar puntos | C: confirmar | R: reiniciar", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return personas, frame_out, movimientos
        
        # ===== MODO NORMAL (después de calibración) =====
        
        # Dibujar línea de cruce
        cv2.line(frame_out, (0, y_cruce), (ancho, y_cruce), (0, 255, 255), 3)
        cv2.putText(frame_out, "LINEA DE CRUCE", (10, y_cruce - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for idp, info in personas.items():
            bbox = info['bbox']
            pies = info['pies']
            clasificacion = info['clasificacion']
            altura_cm = info['altura_cm']
            color = info['color']

            mov = self.detectar_cruce_linea(idp, pies[1])
            if mov:
                movimientos.append({
                    'tipo': mov,
                    'clasificacion': clasificacion,
                    'altura_px': info['altura_px'],
                    'altura_cm': altura_cm
                })

            # Dibujar rectángulo
            cv2.rectangle(frame_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            # Dibujar punto en los pies
            cv2.circle(frame_out, pies, 8, (0, 255, 255), -1)
            
            # Información de la persona
            texto_info = f"ID:{idp} {clasificacion}"
            if altura_cm:
                texto_altura = f"{altura_cm}cm"
            else:
                texto_altura = f"{info['altura_px']}px (sin calibrar)"
            
            # Fondo para el texto
            (w1, h1), _ = cv2.getTextSize(texto_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (w2, h2), _ = cv2.getTextSize(texto_altura, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(frame_out, (bbox[0], bbox[1] - h1 - 35), 
                         (bbox[0] + max(w1, w2) + 10, bbox[1]), (0, 0, 0), -1)
            
            # Textos
            cv2.putText(frame_out, texto_info, (bbox[0] + 5, bbox[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame_out, texto_altura, (bbox[0] + 5, bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Información general en pantalla
        info_altura = 140 if self.calibrado else 120
        cv2.rectangle(frame_out, (10, 10), (350, info_altura), (0, 0, 0), -1)
        
        if self.calibrado:
            cv2.putText(frame_out, f"CALIBRADO: {self.factor_conversion:.4f} cm/px", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            offset_y = 30
        else:
            cv2.putText(frame_out, "SIN CALIBRAR", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            offset_y = 30
        
        cv2.putText(frame_out, f"Entradas: {self.total_entradas}", (20, 35 + offset_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_out, f"Salidas: {self.total_salidas}", (20, 65 + offset_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_out, f"Detectados: {len(personas)}", (20, 95 + offset_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return personas, frame_out, movimientos
    

    # =========================================================
    # === REGISTRO EN BASE DE DATOS ===========================
    # =========================================================

    def registrar_movimiento(self, tipo, camara, altura_px=None, altura_cm=None, clasificacion=None):
        """Registra movimiento con reconexión automática"""
        if not self.verificar_reconectar_db():
            print("✗ No se pudo conectar a la base de datos")
            return False
        
        intentos = 0
        max_intentos = 3
        
        while intentos < max_intentos:
            try:
                fecha = datetime.now()
                self.cursor.execute(
                    """INSERT INTO movimientos 
                       (fecha_hora, tipo_movimiento, camara, altura_pixeles, altura_estimada_cm, clasificacion) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (fecha, tipo, camara, altura_px, altura_cm, clasificacion)
                )

                if tipo == "entrada":
                    self.cursor.execute("UPDATE aforo_actual SET personas_dentro = personas_dentro + 1, total_entradas = total_entradas + 1")
                    self.total_entradas += 1

                elif tipo == "salida":
                    self.cursor.execute("UPDATE aforo_actual SET personas_dentro = GREATEST(personas_dentro - 1, 0), total_salidas = total_salidas + 1")
                    self.total_salidas += 1

                self.conexion_db.commit()
                
                if altura_cm:
                    print(f"[{fecha.strftime('%H:%M:%S')}] {tipo.upper()} - {clasificacion} ({altura_cm}cm)")
                else:
                    print(f"[{fecha.strftime('%H:%M:%S')}] {tipo.upper()} - {clasificacion} ({altura_px}px)")
                return True
                
            except mysql.connector.Error as e:
                print(f"⚠ Error en BD (intento {intentos + 1}/{max_intentos}): {e}")
                intentos += 1
                
                if intentos < max_intentos:
                    time.sleep(1)
                    if not self.verificar_reconectar_db():
                        print("✗ Fallo en reconexión")
                        continue
                else:
                    print("✗ No se pudo registrar el movimiento después de varios intentos")
                    return False
        
        return False


    # =========================================================
    # === EJECUCIÓN ===========================================
    # =========================================================

    def ejecutar(self, fuente_video, nombre_camara="Cámara", posicion_y_cruce=0.5, 
                 altura_referencia_cm=None, modo_calibracion_inicial=False):

        self.posicion_y_cruce = posicion_y_cruce

        if not self.conectar_db():
            print("✗ No se pudo establecer conexión inicial. Saliendo...")
            return
        
        self.crear_tablas()

        print("Cargando modelo YOLO...")
        self.modelo = YOLO("yolov8n.pt")
        print("✓ Modelo cargado")

        # Iniciar calibración si se especificó
        if modo_calibracion_inicial and altura_referencia_cm:
            self.iniciar_calibracion(altura_referencia_cm)

        reader = RTSPReader(fuente_video)
        print("Iniciando cámara RTSP optimizada...")

        # Crear ventana y configurar callback del mouse
        cv2.namedWindow("Sistema de Aforo Inteligente")
        cv2.setMouseCallback("Sistema de Aforo Inteligente", self.mouse_callback_calibracion)

        skip = 0
        ultimo_frame_procesado = None
        contador_sin_db = 0

        while True:
            frame = reader.read()
            if frame is None:
                if ultimo_frame_procesado is not None:
                    cv2.imshow("Sistema de Aforo Inteligente", ultimo_frame_procesado)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                continue

            skip += 1
            if skip % 2 != 0:
                if ultimo_frame_procesado is not None:
                    cv2.imshow("Sistema de Aforo Inteligente", ultimo_frame_procesado)
                else:
                    cv2.imshow("Sistema de Aforo Inteligente", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.modo_calibracion:
                    if self.confirmar_calibracion():
                        self.guardar_calibracion_db(nombre_camara)
                elif key == ord('r') and self.modo_calibracion:
                    self.reiniciar_calibracion()
                
                continue

            personas, frame_proc, movimientos = self.detectar_y_trackear(frame)
            ultimo_frame_procesado = frame_proc

            # Solo registrar movimientos si NO estamos en modo calibración
            if not self.modo_calibracion:
                for mov_info in movimientos:
                    exito = self.registrar_movimiento(
                        mov_info['tipo'],
                        nombre_camara,
                        mov_info['altura_px'],
                        mov_info['altura_cm'],
                        mov_info['clasificacion']
                    )
                    
                    if not exito:
                        contador_sin_db += 1
                        if contador_sin_db > 10:
                            print("⚠ Múltiples fallos de BD. Verifica la conexión.")
                            contador_sin_db = 0
                    else:
                        contador_sin_db = 0

            cv2.imshow("Sistema de Aforo Inteligente", frame_proc)

            # Capturar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.modo_calibracion:
                if self.confirmar_calibracion():
                    self.guardar_calibracion_db(nombre_camara)
            elif key == ord('r') and self.modo_calibracion:
                self.reiniciar_calibracion()

        reader.stop()
        cv2.destroyAllWindows()
        
        # Cerrar conexión DB
        if self.conexion_db and self.conexion_db.is_connected():
            self.cursor.close()
            self.conexion_db.close()
            print("✓ Conexión a BD cerrada correctamente")


# =============================================================
# === FUNCIÓN PRINCIPAL =======================================
# =============================================================

def main():
    config_db = {
        'host': '82.197.82.29',
        'user': 'u659323332_ebano',
        'password': 'Ebano2025*',
        'database': 'u659323332_ebano'
    }
    
    sistema = ControlAforoCruceLinea(config_db)
    
    # URL RTSP Dahua
    USUARIO = 'admin'
    CONTRASENA = 'admin123'
    IP_CAMARA = '192.168.1.108'
    PUERTO_RTSP = '554'
    URL_RTSP_DAHUA = f'rtsp://{USUARIO}:{CONTRASENA}@{IP_CAMARA}:{PUERTO_RTSP}/cam/realmonitor?channel=1&subtype=0'

    # ===== CONFIGURACIÓN DE CALIBRACIÓN =====
    # Opción 1: Iniciar con calibración de puerta
    # La puerta tiene una altura conocida de 210 cm (ajusta según tu caso)
    ALTURA_PUERTA_CM = 210  # Cambia esto a la altura real de tu puerta/referencia
    
    sistema.ejecutar(
        fuente_video=URL_RTSP_DAHUA, 
        nombre_camara='Dahua DH-IPC-HFW1239S1-A-IL',
        posicion_y_cruce=0.7,
        altura_referencia_cm=ALTURA_PUERTA_CM,  # Altura de la puerta en cm
        modo_calibracion_inicial=True  # Iniciar en modo calibración
    )
    
    # ===== INSTRUCCIONES DE USO =====
    # 1. Al iniciar, el sistema entrará en MODO CALIBRACIÓN
    # 2. Haz clic en la PARTE SUPERIOR de la puerta/marco de referencia
    # 3. Haz clic en la PARTE INFERIOR de la puerta/marco de referencia
    # 4. Presiona 'C' para CONFIRMAR la calibración
    # 5. Presiona 'R' para REINICIAR si te equivocaste
    # 6. Una vez calibrado, el sistema calculará automáticamente las alturas reales
    # 7. Presiona 'Q' para salir del sistema


if __name__ == "__main__":
    main()
