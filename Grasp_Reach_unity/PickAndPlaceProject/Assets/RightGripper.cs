using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RightGripper : MonoBehaviour
{
    public robotagent4 agent;

    private void OnTriggerEnter(Collider other)
    {
        if (other.transform.CompareTag("target"))
        {
            agent.RightHit();
            Debug.Log("Left");
        }
        // else if (other.transform.CompareTag("table"))
        // {
        //     agent.TableHit();
        //     Debug.Log("Table");
        // }
        else if (other.transform.CompareTag("robotpart"))
        {
            agent.RobotHit();
            Debug.Log("Robot");
        }
    }

    
}
