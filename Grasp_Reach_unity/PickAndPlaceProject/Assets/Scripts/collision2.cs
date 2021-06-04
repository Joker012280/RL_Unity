using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class collision2 : MonoBehaviour
{

    public robotagent2 agent2;

    
    private void OnTriggerEnter(Collider other)
    {
        if (other.transform.CompareTag("table"))
        {
            agent2.TableHit();
            Debug.Log("Table");
        }
        else if(other.transform.CompareTag("robotpart"))
        {
            
            agent2.TableHit();
            Debug.Log("Robot Itself");
        }
        
    }
}
