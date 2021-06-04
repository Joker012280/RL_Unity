using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent3Colli : MonoBehaviour
{
    public robotagent3 agent;

    
    private void OnTriggerEnter(Collider other)
    {
        if (other.transform.CompareTag("table"))
        {
            agent.TableHit();
            Debug.Log("Table");
        }
        else if(other.transform.CompareTag("robotpart"))
        {
            
            agent.TableHit();
            Debug.Log("Robot Itself");
        }
        
    }
}
