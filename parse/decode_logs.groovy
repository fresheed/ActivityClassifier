import org.fresheed.actionlogger.utils.EventsLogCompressor
import org.fresheed.actionlogger.events.ActionEvent
import org.fresheed.actionlogger.events.ActionLog
import java.nio.file.Files

File bin_file=new File(args[0])
byte[] raw_bytes = Files.readAllBytes(bin_file.toPath())

compressor=new EventsLogCompressor()
ActionLog log=compressor.decompressEventsLog(raw_bytes, 3)

for (ActionEvent event: log.getEvents()){
    print "${event.getTimestamp()} "
    event.getValues().each {
       print "${it} "
    }      
    println ""
}