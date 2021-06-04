using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class rightcollision : MonoBehaviour
{
    public robotagent3 agent;

    private void OnTriggerEnter(Collider other)
    {
        if (other.transform.CompareTag("table"))
        {
            agent.TableHit();
            Debug.Log("Table");
        }
        else if (other.transform.CompareTag("robotpart"))
        {
            agent.RobotHit();
            Debug.Log("Robot");
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.transform.CompareTag("target"))
        {
            agent.RightHit();
            Debug.Log("Right");
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if(other.transform.CompareTag("target"))
        {
            agent.RightHitCancel();
            Debug.Log("Right Canceled");
        }        
    }

}
