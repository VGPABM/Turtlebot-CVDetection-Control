#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib as plt
from pyzbar.pyzbar import decode


import imutils
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import sys


robot_x = 0

vid= cv2.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)


def pose_callback(pose):
    global robot_x
    global robot_y
    global robot_t

    rospy.loginfo("Robot x = %f\n RObot Y = %f\n Robot T = %f", pose.x, pose.y, pose.theta)
    robot_x = pose.x  # menyimpan float posisi turtlesim pada sumbu x
    robot_y = pose.y  # menyimpan float posisi turtlesim pada sumbu y
    robot_t = pose.theta  # menyimpan float posisi derajat turtlesim



def hijau(lin_vel, ang_vel):
    rospy.init_node('move_turtle', anonymous=False)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist,
                          queue_size=10)

    rospy.Subscriber('/turtle1/pose', Pose,
                     pose_callback)

    rate = rospy.Rate(10)

    vel = Twist()

    vel.linear.x = lin_vel
    vel.linear.y = 0
    vel.linear.z = 0

    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = 0

    pub.publish(vel)
    rate.sleep()

def merah(lin_vel, ang_vel):
    rospy.init_node('move_turtle', anonymous=False)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist,
                          queue_size=10)

    rospy.Subscriber('/turtle1/pose', Pose,
                     pose_callback)

    rate = rospy.Rate(10)
    vel = Twist()


    vel.linear.x = 0
    vel.linear.y = 0
    vel.linear.z = 0

    vel.angular.x = 0
    vel.angular.y = 0
    vel.angular.z = 0
    pub.publish(vel)
    rate.sleep()

def kiri(lin_vel, ang_vel):
    rospy.init_node('move_turtle', anonymous=False)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist,
                          queue_size=10)

    rospy.Subscriber('/turtle1/pose', Pose,
                     pose_callback)

    rate = rospy.Rate(10)

    vel = Twist()

    posisilama = pose.theta

    while True:
        vel.linear.x = 0
        vel.linear.y = 0
        vel.linear.z = 0

        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = ang_vel
        if abs(robot_t - posisilama) == 1.5708:
            return

    pub.publish(vel)
    rate.sleep()

def kanan(lin_vel, ang_vel):
    rospy.init_node('move_turtle', anonymous=False)
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist,
                          queue_size=10)

    rospy.Subscriber('/turtle1/pose', Pose,
                     pose_callback)

    rate = rospy.Rate(10)

    vel = Twist()

    posisilama = pose.theta

    while True:
        vel.linear.x = 0
        vel.linear.y = 0
        vel.linear.z = 0

        vel.angular.x = 0
        vel.angular.y = 0
        vel.angular.z = ang_vel
        if abs(robot_t-posisilama) == 1.5708:
            return

    pub.publish(vel)
    rate.sleep()

if __name__ == '__main__':
    try:
        while True:
            ret, frame = vid.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = np.array([136, 87, 111])
            upper_red = np.array([180, 255, 255])

            lower_yellow = np.array([0, 0, 0])
            upper_yellow = np.array([45, 255, 255])

            lower_green = np.array([50, 50, 72])
            upper_green = np.array([70, 255, 255])

            lower_blue = np.array([94, 80, 2])
            upper_blue = np.array([120, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            mask2 = cv2.inRange(hsv, upper_yellow, upper_yellow)
            mask3 = cv2.inRange(hsv, lower_green, upper_green)
            mask4 = cv2.inRange(hsv, lower_blue, upper_blue)

            cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts1 = imutils.grab_contours(cnts1)

            cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts2 = imutils.grab_contours(cnts2)

            cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts3 = imutils.grab_contours(cnts3)

            cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts4 = imutils.grab_contours(cnts4)

            for c in cnts1:
                area = cv2.contourArea(c)
                if area > 5000:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

                    M = cv2.moments(c)

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "Merah", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
                    merah(0.0,0.0)

            for c in cnts2:
                area = cv2.contourArea(c)
                if area > 5000:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

                    M = cv2.moments(c)

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    hijau(2.0, 0.0)

                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "Kuning", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

            for c in cnts3:
                area = cv2.contourArea(c)
                if area > 5000:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

                    M = cv2.moments(c)

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    hijau(3.0,0.0)

                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "Hijau", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

            for c in cnts4:
                area = cv2.contourArea(c)
                if area > 5000:
                    cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

                    M = cv2.moments(c)

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "Biru", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

            code = decode(frame)
            for barcode in decode(frame):
                myData = barcode.data.decode('utf-8')
                pts = np.array([barcode.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 127, 255), 5)
                print(myData)
                pts2 = barcode.rect
                cv2.putText(frame, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 127, 255), 2)

                if myData == 'kiri':
                    kiri(0.0,0.5)
                elif myData == 'kanan':
                    kanan(0.0,-0.5)


            cv2.imshow("result", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass