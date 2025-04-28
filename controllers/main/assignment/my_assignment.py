import numpy as np
import time
import cv2
#import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

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
    if sensor_data['z_global'] < 0.2:
        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
        return control_command

    if not hasattr(get_command, "counter"):
        get_command.counter = 0
    if not hasattr(get_command, "actual_info"):
        get_command.actual_info = None
    if not hasattr(get_command, "previous_info"):
        get_command.previous_info = None
    if not hasattr(get_command, "goal_reached"):
        get_command.goal_reached = True
    if not hasattr(get_command, "last_goal"):
        get_command.last_goal = None
    if not hasattr(get_command, "goals_list"):
        get_command.goals_list = []
    if not hasattr(get_command, "mode"):
        get_command.mode = "exploration"
    if not hasattr(get_command, "start_position"):
        get_command.start_position = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

    TOL = 0.1
    GOAL_SPACEMENT = 1.0

    pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

    get_command.counter += 1
    if get_command.counter % 10 == 0:
        get_command.counter = 0
        # Sauvegarder l'ancienne info
        get_command.previous_info = get_command.actual_info
        get_command.actual_info = {
            "camera_image": camera_data,
            "drone_position": pos,
            "drone_quaternion": np.array([sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']])
        }

        if get_command.mode == "exploration" and get_command.previous_info is not None and get_command.goal_reached:
            goal, yaw_correction = findGoal(get_command.previous_info, get_command.actual_info)
            get_command.last_yaw_correction = yaw_correction

            if goal is not None and np.all(np.isfinite(goal)):
                direction_vector = goal - pos
                norm = np.linalg.norm(direction_vector)
                if norm != 0:
                    extension_vector = 0.2 * direction_vector / norm  # 20cm derrière la gate
                else:
                    extension_vector = np.array([0, 0, 0])
                    goal = goal + extension_vector

                updated = False
                idx = 0
                # is_new = True
                for saved_goal in get_command.goals_list:
                    dist_between_goals = goal - saved_goal
                    if (np.abs(dist_between_goals[0]) < GOAL_SPACEMENT) and (np.abs(dist_between_goals[1]) < GOAL_SPACEMENT) and (np.abs(dist_between_goals[2]) < GOAL_SPACEMENT):
                        get_command.goals_list[idx] = goal
                        print('Goal updated with new position: ', goal)
                        updated = True
                        break
                    idx += 1

                if not updated:
                    get_command.goals_list.append(goal)
                    print('New goal stored: ', goal)

                get_command.last_goal = goal
                get_command.goal_reached = False

    if get_command.mode == "exploration":
        if get_command.last_goal is not None:
            dist = pos - get_command.last_goal

            adjusted_yaw = sensor_data['yaw'] + get_command.last_yaw_correction
            #desired_yaw = np.arctan2(get_command.last_goal[1] - pos[1], get_command.last_goal[0] - pos[0])
            #yaw_error = desired_yaw - sensor_data['yaw']
            #yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

            #YAW_TOL = 0.1

            if (np.abs(dist[0])< TOL) and (np.abs(dist[1]) < TOL) and (np.abs(dist[2]) < TOL):
                print("Goal reached!, distance: ", dist)
                get_command.goal_reached = True
                control_command = [pos[0], pos[1], pos[2], sensor_data['yaw']]
                get_command.last_goal = None

            #elif np.abs(yaw_error) > YAW_TOL:
            #    # Si on n'est pas encore bien orienté : corriger le yaw
            #    YAW_CORRECTION_SPEED = 0.3  # rad/s, ajuste si nécessaire
            #    control_command = [pos[0], pos[1], pos[2], sensor_data['yaw'] + np.sign(yaw_error) * YAW_CORRECTION_SPEED]

            else:
                control_command = [get_command.last_goal[0], get_command.last_goal[1], get_command.last_goal[2], sensor_data['yaw']]
        else:
            # Drift to search
            DRIFT_SPEED = 0.02
            DRIFT_SPEED_Z = 0.01
            YAW_STEP = 0.05
            control_command = [pos[0] + DRIFT_SPEED, pos[1] + DRIFT_SPEED, pos[2] + DRIFT_SPEED_Z, sensor_data['yaw'] + YAW_STEP]

        if len(get_command.goals_list) >= 5 and get_command.goal_reached:
            #print('voici les goals trouvés: ', get_command.goals_list)
            print("5 goals found. Switching to navigate mode.")
            get_command.mode = "navigate"
            get_command.goals_list = reorder_goals(get_command.goals_list)
            #print('liste réarrangée: ', get_command.goals_list)
            #plot_goals(get_command.goals_list)
            get_command.last_goal = get_command.start_position

    elif get_command.mode == "navigate":
        if not hasattr(get_command, "trajectory_idx"):
            # Initialisation à l'entrée du mode navigate
            get_command.trajectory = generate_trajectory(get_command.goals_list, get_command.start_position, points_per_segment=20)
            get_command.trajectory_idx = 0

        MARGIN = 0.3

        # Récupérer les points
        trajectory = get_command.trajectory
        idx = get_command.trajectory_idx

        if idx < len(trajectory) - 2:
            target_point = trajectory[idx + 2]
        else:
            target_point = trajectory[-1]  # Rester au dernier point pour éviter l'overflow

        actual_point = trajectory[idx]

        # Calcul de la distance actuelle au point actuel (pas au target +2)
        dist_before_change = np.abs(pos - actual_point)

        if (dist_before_change[0] < MARGIN) and (dist_before_change[1] < MARGIN) and (dist_before_change[2] < MARGIN):
            print(f"Reached trajectory point {idx+1}/{len(trajectory)}")
            get_command.trajectory_idx += 1  # Avancer au point suivant

            # Vérifier qu'on ne dépasse pas la trajectoire
            if get_command.trajectory_idx >= len(trajectory):
                print("Full trajectory completed! Drone stays at last point.")
                get_command.trajectory_idx = len(trajectory) - 1  # Bloquer sur le dernier point

        # Toujours viser target_point
        control_command = [target_point[0], target_point[1], target_point[2], sensor_data['yaw']]

    return control_command

def generate_trajectory(goals_list, start_position, points_per_segment=10):
    """
    Génère une trajectoire douce (spline) passant 2x par les gates, en partant du point de départ et retournant au départ.

    Args:
        goals_list (list of np.array): Liste des positions [x, y, z] des gates.
        start_position (np.array): Position de départ [x, y, z].
        points_per_segment (int): Nombre de points interpolés entre chaque paire de points.

    Returns:
        trajectory (list of np.array): Liste de positions interpolées.
    """

    # Construction de la séquence : départ -> gates 2x -> retour départ
    sequence = [start_position.copy()]  # 🔵 D'abord le départ
    for _ in range(2):
        for goal in goals_list:
            sequence.append(goal.copy())
    sequence.append(start_position.copy())  # 🔵 Retour final au départ

    sequence = np.array(sequence)  # Convertir en numpy array

    # Créer un paramètre "temps" t pour les points clé
    t = np.linspace(0, 1, len(sequence))

    # Spline interpolation pour chaque coordonnée
    cs_x = CubicSpline(t, sequence[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, sequence[:, 1], bc_type='natural')
    cs_z = CubicSpline(t, sequence[:, 2], bc_type='natural')

    # Générer les points interpolés
    t_fine = np.linspace(0, 1, points_per_segment * (len(sequence) - 1))
    trajectory = []
    for ti in t_fine:
        x = cs_x(ti)
        y = cs_y(ti)
        z = cs_z(ti)
        trajectory.append(np.array([x, y, z]))

    return trajectory

def reorder_goals(goals_list):
    """
    Trie les goals dans le sens trigonométrique autour du centre, 
    en commençant par celui en bas à droite.
    """
    if len(goals_list) != 5:
        print("Attention: nombre de goals différent de 5")
        return goals_list

    goals_array = np.array(goals_list)
    center = np.mean(goals_array[:, :2], axis=0)  # Moyenne en x, y

    # Calculer l'angle de chaque goal par rapport au centre
    angles = np.arctan2(goals_array[:,1] - center[1], goals_array[:,0] - center[0])

    # Ordonner les angles dans le sens trigonométrique (croissant)
    sorted_indices = np.argsort(angles)

    # Reorganiser les goals
    sorted_goals = goals_array[sorted_indices]

    # Maintenant, trouver celui le plus en bas à droite
    bottom_right_index = np.argmax(sorted_goals[:,0] + sorted_goals[:,1])

    # Décaler pour commencer par celui du bas à droite
    sorted_goals = np.roll(sorted_goals, -bottom_right_index, axis=0)

    return list(sorted_goals)

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

##### Fonction debeug : #####

def plot_goals(goals_list):
    """
    Affiche la liste des goals en 2D avec matplotlib pour vérifier l'ordre.
    """
    goals_array = np.array(goals_list)
    
    plt.figure()
    plt.plot(goals_array[:,0], goals_array[:,1], 'o-', markersize=8, label='Goals Path')
    
    for idx, (x, y, _) in enumerate(goals_array):
        plt.text(x, y, str(idx+1), fontsize=12, ha='right', va='bottom')  # Numérote les points
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Reordered Goals Path')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()
