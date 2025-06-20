package furhatos.app.faceskill

import furhatos.app.faceskill.flow.main.Main
import furhatos.app.faceskill.flow.main.Emotion
import furhatos.event.EventSystem
import furhatos.skills.Skill
import furhatos.flow.kotlin.*
import furhatos.util.CommonUtils
import org.zeromq.ZMQ
import kotlinx.coroutines.launch
import kotlinx.coroutines.GlobalScope
import zmq.ZMQ.ZMQ_SUB

val logger = CommonUtils.getRootLogger()
val objserv = "tcp://<your server ip>:9999" //The TCP socket of the object server

val subSocket: ZMQ.Socket = getConnectedSocket(ZMQ_SUB, objserv) //Makes a socket of the object server
val emotion = "emotion_"

/**
 * Parses a message from the object server, turns the message into a list of objects.
 */

fun getEmotion(message: String, delimiter: String): List<String> {
    val emotion = mutableListOf<String>()
    message.split(" ").forEach {
        if(it.startsWith(delimiter)) {
            emotion.add(it.removePrefix(delimiter))
        }
    }
    return emotion
}

/**
 * Function that starts a thread which continuously polls the object server.
 * Based on what is in the message will either send:
 *These events can be caught in the flow (Main), and be responded to.
 */
fun startListenThread() {
    GlobalScope.launch { // launch a new coroutine in background and continue
        logger.warn("LAUNCHING COROUTINE")
        subSocket.subscribe("")
        while (true) {
            val message = subSocket.recvStr()
            logger.warn("got: $message")
            if(message.contains(emotion)){
                EventSystem.send(Emotion(getEmotion(message, emotion)))
            }
        }
    }
}

class FaceSkill : Skill() {
    override fun start() {
        startListenThread()
        Flow().run(Main)
    }
}

fun main(args: Array<String>) {
    logger.warn("starting main")
    Skill.main(args)
}