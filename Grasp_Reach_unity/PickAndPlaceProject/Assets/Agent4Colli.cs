using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent4Colli : MonoBehaviour
{
    public robotagent4 agent;

    private void OnTriggerEnter(Collider other)
    {
        
        if (other.transform.CompareTag("table"))
        {
            agent.TableHit();
            Debug.Log("Table");
        }
        
    }
    
}
