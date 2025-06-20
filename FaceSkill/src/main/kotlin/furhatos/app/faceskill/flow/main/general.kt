package furhatos.app.faceskill.flow.main

import furhatos.app.faceskill.getConnectedSocket
import furhatos.app.faceskill.objserv
import furhatos.event.Event
import furhatos.flow.kotlin.furhat
import furhatos.flow.kotlin.state
import furhatos.flow.kotlin.*
import furhatos.util.CommonUtils
import org.zeromq.ZMQ
import zmq.ZMQ.ZMQ_PUB


/**
 * Events used for communication between the thread and the flow.
 */
val OUT_PORT = "10.20.3.150:8888"
class Emotion(val emotion: List<String>): Event()
val pubSocket: ZMQ.Socket = getConnectedSocket(ZMQ_PUB, OUT_PORT)

/**
 * Main flow that starts the camera feed and awaits events sent from the thread
 */



val Main = state {


    onEntry {
        furhat.cameraFeed.enable()
        furhat.ask("Welcome to Jacksons experiment. testing stages. How are you feeling today?")
        furhat.listen(timeout = 60000)
    }

    onUserEnter {
        furhat.attend(it)
        furhat.listen(timeout = 60000)
    }

    onResponse {
        val userText = it.text
        pubSocket.send("speech_$userText")
        furhat.listen(timeout = 60000)
    }


    onEvent<Emotion>{
        val user = users.current
        if (user.isAttendingFurhat()) {
            furhat.ask("you look like you are feeling ${it.emotion}")
        }
        furhat.listen(timeout = 60000)
    }

    onExit {
        furhat.cameraFeed.disable()
    }
}