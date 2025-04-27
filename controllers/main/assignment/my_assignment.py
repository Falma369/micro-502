import numpy as np
import time
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors. 
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

def get_command(sensor_data, camera_data, dt):
    # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
    # If you want to display the camera image you can call it main.py.

    # Take off example
    if sensor_data['z_global'] < 0.2:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    if not hasattr(get_command, "lost_counter"):
        get_command.lost_counter = 0
    if not hasattr(get_command, "search_yaw"):
        get_command.search_yaw = 0.0
    if not hasattr(get_command, "goal_reached"):
        get_command.goal_reached = False
    if not hasattr(get_command, "counter"):
        get_command.counter = 0
    if not hasattr(get_command, "actual_info"):
        get_command.actual_info = None
    if not hasattr(get_command, "last_goal"):
        get_command.last_goal = None
    if not hasattr(get_command, "last_yaw_correction"):
        get_command.last_yaw_correction = 0.0
    if not hasattr(get_command, "goals_list"):
        get_command.goals_list = []
    if not hasattr(get_command, "mode"):
        get_command.mode = "exploration"  # "exploration" or "navigate"
    if not hasattr(get_command, "start_position"):
        get_command.start_position = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

    TOL = 0.15

    get_command.counter += 1
    if ((get_command.counter % 50) == 0):
        get_command.counter = 0
        previous_info = get_command.actual_info
        #if (previous_info is not None):
        #    cv2.imshow("prev image: ", previous_info["camera_image"])
        get_command.actual_info = {
            "camera_image" : camera_data,
            "drone_position" : np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]),
            "drone_quaternion" : np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
        }
        #cv2.imshow("actual image: ", get_command.actual_info["camera_image"])

        if previous_info is not None and get_command.mode == "exploration":
            
            goal, yaw_correction = findGoal(previous_info, get_command.actual_info)
            get_command.last_yaw_correction = yaw_correction
            if goal is not None and np.all(np.isfinite(goal)):
                get_command.last_goal = goal
                get_command.lost_counter = 0
                get_command.goal_reached = False         # on a une nouvelle cible
            
                MARGIN = 2

                is_new = True
                for saved_goal in get_command.goals_list:
                    if np.linalg.norm(goal - saved_goal) < MARGIN:
                        is_new = False
                        break

                if is_new:
                    get_command.goals_list.append(goal)
                    print('f New goal found: ', goal)
                    print('goal list : ', get_command.goals_list)
                else:
                    print('Goal too close to existing one, not stored')
            if len(get_command.goals_list) >= 5:
                print("5 goals found! Switching to navigate mode.")
                get_command.mode = "navigate"
                get_command.last_goal = get_command.start_position
                get_command.goal_reached = False
            
            else:
                get_command.lost_counter += 1

    SEARCH_MODE = (
        get_command.goal_reached and             # on est arrivé
        get_command.lost_counter >= 10           # …et on ne voit plus rien
    )

    pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
    
    if get_command.mode == "exploration": 
        # Mode recherche des goals
        if get_command.last_goal is not None:
            dist_to_goal = pos - get_command.last_goal
            print ('dist_to_goal= ', dist_to_goal)
            if (np.abs(pos[0] - get_command.last_goal[0]) < TOL) and (np.abs(pos[1] - get_command.last_goal[1]) < TOL) and (np.abs(pos[2] - get_command.last_goal[2]) < TOL):
                get_command.goal_reached = True
                print('goal atteint')

        if get_command.last_goal is not None and not SEARCH_MODE:      
            adjusted_yaw = sensor_data['yaw'] + get_command.last_yaw_correction
            control_command = [get_command.last_goal[0], get_command.last_goal[1], get_command.last_goal[2], adjusted_yaw]
        # if (get_command.lost_drone == 1):
        #     print('pas d obstacle en vue, je tourne')
        #     control_command = [get_command.last_goal[0], get_command.last_goal[1], get_command.last_goal[2], adjusted_yaw]
        else:
            DRIFT_SPEED = 0.05
            DRIFT_SPEED_Z = 0.01
            YAW_STEP    = 0.05
            control_command = [pos[0] + DRIFT_SPEED, pos[1] + DRIFT_SPEED, pos[2] + DRIFT_SPEED_Z, sensor_data['yaw'] + YAW_STEP]

    elif get_command.mode == "navigate":
        # Mode navigation
        dist_to_start = np.linalg.norm(pos - get_command.start_position)
        if (np.abs(pos[0] - dist_to_start[0]) < TOL) and (np.abs(pos[1] - dist_to_start[1]) < TOL) and (np.abs(pos[2] - dist_to_start[2]) < TOL):
            print("Back to start position, ready to start navigation!")
            control_command = [pos[0], pos[1], pos[2], sensor_data['yaw']]
        else:
            control_command = [get_command.start_position[0], get_command.start_position[1], get_command.start_position[2], sensor_data['yaw']]


    return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

def save_image(camera_data, x_global, y_global, z_global, q_x, q_y, q_z, q_w):
    previous_image["camera_image"]: camera_data
    previous_image["drone_position"]: np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]) 
    previous_image["drone_quaternion"]: np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
    return

def detect_trapezes(picture):
    """Détecte les trapèzes roses dans l'image et retourne leurs coordonnées."""
    
    # Convertir en format NumPy si nécessaire
    image = np.array(picture, dtype=np.uint8)

    # Convertir l'image en HSV (meilleur pour filtrer les couleurs)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir la plage de couleur rose en HSV
    lower_pink = np.array([140, 50, 50])  # Ajuster si nécessaire
    upper_pink = np.array([170, 255, 255])
    
    # Appliquer un masque pour isoler les zones roses
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    trapezes = []  # Liste des trapèzes détectés (coins + centre)
    trapezes_centre = []  # Liste des centres des trapèzes

    for contour in contours:
        # Approximer la forme pour détecter les polygones
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Un trapèze a 4 sommets
            # Calculer le centre
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            trapezes.append({"coins": approx.reshape(4, 2), "centre": np.array([cX, cY])})
            #trapezes_centre.append((cX, cY))  # Ajout du centre uniquement

            # Dessiner le trapèze détecté
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

    # Afficher l'image traitée (seulement si Webots le permet)
    cv2.imshow("Trapezes Detection", image)
    cv2.waitKey(1)  # Mettre une pause pour éviter les crashs

    return trapezes  # Retourne les coordonnées des centres des trapèzes trouvés

def select_big_trapeze(trapezes):
    """
    Sélectionne le plus grand trapèze (en surface) dans la liste
    et retourne les coordonnées de son centre.
    """
    if (len(trapezes) == 0):
        return None  # Aucun trapèze détecté

    max_area = 0
    biggest = None

    for trapeze in trapezes:
        # Calculer l'aire du polygone
        contour = trapeze["coins"]
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest = trapeze
        
    if biggest is None:
        return None

    # Transform the center
    x_c, y_c = biggest["centre"]
    x_c = x_c - 150
    y_c = -y_c + 150
    biggest["centre"] = (x_c, y_c)

    # Transform the corners
    transformed_corners = []
    for (x, y) in biggest["coins"]:
        new_x = x - 150
        new_y = -y + 150
        transformed_corners.append((new_x, new_y))

    biggest["coins"] = np.array(transformed_corners)

    # Calculer correction de yaw : distance du centre du trapèze par rapport au centre de l’image
    image_center_x = 150  # Si ton image fait 300px de large
    delta_x = x_c + 150 - image_center_x  # Revenir aux coords originales (avant transformation)
    delta_x_norm = -delta_x / image_center_x  # Normalisé entre [-1, 1]
    
    biggest["yaw_correction"] = delta_x_norm

    return biggest

def findGoal(previous_info, actual_info):
    # Détecter les trapèzes dans l'image

    trapezes = detect_trapezes(previous_info["camera_image"])
    if trapezes is None:
        return None, 0.0

    gate_prev = select_big_trapeze(trapezes)
    if gate_prev is None:
        return None, 0.0
    vect1 = vect_ToPoint(previous_info["drone_quaternion"], gate_prev["centre"])

    trapezes = detect_trapezes(actual_info["camera_image"])
    gate = select_big_trapeze(trapezes)
    if gate is None:
        return None, 0.0
    vect2 = vect_ToPoint(actual_info["drone_quaternion"], gate["centre"])

    goal = triangulation (vect1, vect2, previous_info["drone_position"], actual_info["drone_position"])
    
    yaw_correction = gate["yaw_correction"]

    return goal, yaw_correction

def vect_ToPoint(quat, point_img):
    focal_length = (300/(2*np.tan(1.5/2)))
    euler_angles = np.array([np.pi/2, np.pi/2, 0])

    pos_d = np.array([focal_length, -point_img[0], point_img[1]])

    R_d2w = (quaternion2rotmat(quat))

    vect_direction = R_d2w @ pos_d
    vect_direction = vect_direction / np.linalg.norm(vect_direction)

    return vect_direction

def triangulation (vect1, vect2, pos1, pos2):
    A = ([[vect1[0], -vect2[0]],
          [vect1[1], -vect2[1]],
          [vect1[2], -vect2[2]]])
    
    b = ([[pos2[0] - pos1[0]],
          [pos2[1] - pos1[1]],
          [pos2[2] - pos1[2]]])

    lambda_mu, *_ = np.linalg.lstsq(A, b)
    lamb = lambda_mu[0].item()
    mu = lambda_mu[1].item()

    P1 = pos1 + lamb * vect1
    P2 = pos2 + mu * vect2

    OffsetDroneCam = np.array([0.03, 0, 0.01])

    P = (P1 + P2)/2 + OffsetDroneCam

    return P

def euler2rotmat(euler_angles):
    
    R = np.eye(3)
    
    # rotation matrix
    # Rotation matrix for each angle (roll, pitch, yaw)
    # Multiply the matrices together to get the total rotation matrix

    # Inputs:
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    # Outputs:
    #           R: A 3x3 numpy array that represents the rotation matrix of the euler angles
    

    # --- SAMPLE SOLUTION ---

    R_roll = np.array([ [1, 0, 0, 0], 
                        [0, np.cos(euler_angles[0]), -np.sin(euler_angles[0]), 0],
                        [0, np.sin(euler_angles[0]), np.cos(euler_angles[0]), 0],
                        [0, 0, 0, 1]])
    
    R_pitch = np.array([[np.cos(euler_angles[1]), 0, np.sin(euler_angles[1]), 0],
                        [0, 1, 0, 0],
                        [-np.sin(euler_angles[1]), 0, np.cos(euler_angles[1]), 0],
                        [0, 0, 0, 1]])
    
    R_yaw = np.array([ [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0, 0],
                       [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    
    R = R_yaw @ R_pitch @ R_roll

    return R

def quaternion2rotmat(quaternion):
    
    R = np.eye(3)
    
    # Here you need to calculate the rotation matrix from a quaternion

    # Inputs:
    #           quaternion: A list of 4 numbers [x, y, z, w] that represents the quaternion
    # Outputs:
    #           R: A 3x3 numpy array that represents the rotation matrix of the quaternion

    x, y, z, w = quaternion
    
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                    [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                    [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    
    return R
        