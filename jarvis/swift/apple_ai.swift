// apple-ai: small helper that gives Jarvis (Rust) access to Apple's on-device frameworks, which
// have Swift-only APIs.
//
//   apple-ai status                    JSON: on-device model and Image Playground availability
//   apple-ai chat [INSTRUCTIONS]       chat loop on stdin/stdout, one JSON object per line:
//                                        in : {"prompt": "...", "reset": false}
//                                        out: {"delta": "..."} ... {"done": true}
//                                             or {"error": "...", "code": "unavailable|context|guardrail|refusal|other"}
//                                        in : {"route": "question"}      does it need live/recent data?
//                                        out: {"needs_cloud": true|false} (guided generation, always valid)
//   apple-ai see IMAGE                 Vision: text (OCR), scene labels, faces, animals, barcodes -> JSON
//   apple-ai imagine PROMPT OUT.png [illustration|animation|sketch]
//                                      Image Playground: generate an image on device -> JSON
//
// Build: xcrun swiftc -O -parse-as-library -o rust/target/apple-ai jarvis/swift/apple_ai.swift

import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import ImagePlayground
import UniformTypeIdentifiers
import Vision

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

// ---- chat ----

func errorCode(_ error: Error) -> String {
    guard let e = error as? LanguageModelError else { return "other" }
    switch e {
    case .contextSizeExceeded: return "context"
    case .guardrailViolation: return "guardrail"
    case .refusal: return "refusal"
    case .rateLimited: return "busy"
    default: return "other"
    }
}

/// Routing decision for the assistant. Guided generation makes the model fill this struct, so
/// the answer is a real Bool instead of free text that a small model gets wrong.
@Generable
struct Route {
    @Guide(description: "True only if a correct answer depends on information newer than the model's training"
        + " or on live data: today's news, weather, prices, stock quotes, sports results, traffic, or what is"
        + " happening right now. False for general knowledge, history, science, math, language, jokes, advice,"
        + " small talk, and anything about text the user supplies.")
    var needsLiveData: Bool
}

let routeInstructions = "You decide whether a voice assistant's request needs live or recent information from the internet."

func route(_ question: String) async -> Bool {
    // A fresh session each time: routing must not be colored by the conversation.
    let session = LanguageModelSession(instructions: routeInstructions)
    do {
        let r = try await session.respond(to: "Request: \(question)", generating: Route.self)
        return r.content.needsLiveData
    } catch {
        return false
    }
}

func chat(instructions: String) async {
    let model = SystemLanguageModel.default
    guard case .available = model.availability else {
        emit(["error": "on-device model unavailable: \(model.availability)", "code": "unavailable"])
        exit(2)
    }
    var session = LanguageModelSession(instructions: instructions)
    do {
        for try await line in FileHandle.standardInput.bytes.lines {
            guard let data = line.data(using: .utf8),
                  let req = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                emit(["error": "bad request line", "code": "other"])
                continue
            }
            if let question = req["route"] as? String {
                emit(["needs_cloud": await route(question)])
                continue
            }
            guard let prompt = req["prompt"] as? String else {
                emit(["error": "request needs prompt or route", "code": "other"])
                continue
            }
            if req["reset"] as? Bool == true {
                session = LanguageModelSession(instructions: instructions)
                if req["prompt"] == nil {
                    emit(["done": true])
                    continue
                }
            }
            var attempt = 0
            while true {
                attempt += 1
                var sent = ""
                do {
                    let stream = session.streamResponse(to: prompt)
                    for try await snapshot in stream {
                        let full = snapshot.content
                        if full.hasPrefix(sent) {
                            emit(["delta": String(full.dropFirst(sent.count))])
                        } else {
                            emit(["delta": full])
                        }
                        sent = full
                    }
                    emit(["done": true])
                    break
                } catch {
                    let code = errorCode(error)
                    if code == "context" && attempt == 1 {
                        // Conversation outgrew the model's context window: forget it and retry once.
                        session = LanguageModelSession(instructions: instructions)
                        continue
                    }
                    emit(["error": "\(error)", "code": code])
                    // Do not let a failed turn poison the conversation.
                    session = LanguageModelSession(instructions: instructions)
                    break
                }
            }
        }
    } catch {
        fail("stdin: \(error)")
    }
}

// ---- see ----

func see(path: String) {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("cannot read image \(path)")
    }
    let ocr = VNRecognizeTextRequest()
    ocr.recognitionLevel = .accurate
    ocr.usesLanguageCorrection = true
    ocr.recognitionLanguages = ["en-US", "ko-KR", "ja-JP", "zh-Hans"]
    let classify = VNClassifyImageRequest()
    let faces = VNDetectFaceRectanglesRequest()
    let animals = VNRecognizeAnimalsRequest()
    let barcodes = VNDetectBarcodesRequest()
    do {
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([ocr, classify, faces, animals, barcodes])
    } catch {
        fail("vision: \(error)")
    }
    // Text in reading order: top to bottom (Vision's y axis points up), then left to right.
    let lines = (ocr.results ?? [])
        .sorted { a, b in
            let (ya, yb) = (a.boundingBox.midY, b.boundingBox.midY)
            return abs(ya - yb) > 0.01 ? ya > yb : a.boundingBox.minX < b.boundingBox.minX
        }
        .compactMap { $0.topCandidates(1).first?.string }
    let labels = (classify.results ?? [])
        .filter { $0.confidence > 0.3 }
        .prefix(10)
        .map { ["name": $0.identifier, "confidence": Double($0.confidence)] as [String: Any] }
    let animalNames = (animals.results ?? []).compactMap { $0.labels.first?.identifier }
    let codes = (barcodes.results ?? []).compactMap { $0.payloadStringValue }
    emit([
        "width": image.width, "height": image.height,
        "text": lines, "labels": labels,
        "faces": faces.results?.count ?? 0,
        "animals": animalNames,
        "barcodes": codes,
    ])
}

// ---- imagine ----

func imagine(prompt: String, output: String, styleName: String) async {
    let style: ImagePlaygroundStyle
    switch styleName {
    case "animation": style = .animation
    case "sketch": style = .sketch
    default: style = .illustration
    }
    do {
        let t0 = Date()
        let creator = try await ImageCreator()
        for try await image in creator.images(for: [.text(prompt)], style: style, limit: 1) {
            let url = URL(fileURLWithPath: output)
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                fail("cannot write \(output)")
            }
            CGImageDestinationAddImage(dest, image.cgImage, nil)
            CGImageDestinationFinalize(dest)
            emit(["done": true, "path": output, "width": image.cgImage.width, "height": image.cgImage.height,
                  "seconds": Date().timeIntervalSince(t0)])
            return
        }
        emit(["error": "no image was produced", "code": "other"])
    } catch let e as ImageCreator.Error {
        // notSupported: Image Playground is not available on this Mac (Apple Intelligence off, model
        // not downloaded, or unsupported region/language). Jarvis then falls back to FLUX on MLX.
        emit(["error": "Image Playground: \(e)", "code": "\(e)"])
    } catch {
        emit(["error": "Image Playground: \(error)", "code": "other"])
    }
}

// ---- main ----

@main
struct AppleAI {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        switch args.first {
        case "status":
            var playground = "available"
            do { _ = try await ImageCreator() } catch { playground = "\(error)" }
            emit(["foundation_models": "\(SystemLanguageModel.default.availability)", "image_playground": playground,
                  "macos": ProcessInfo.processInfo.operatingSystemVersionString])
        case "chat":
            await chat(instructions: args.count > 1 ? args[1] : "You are a helpful assistant. Answer briefly.")
        case "see":
            guard args.count > 1 else { fail("usage: apple-ai see IMAGE") }
            see(path: args[1])
        case "imagine":
            guard args.count > 2 else { fail("usage: apple-ai imagine PROMPT OUT.png [illustration|animation|sketch]") }
            await imagine(prompt: args[1], output: args[2], styleName: args.count > 3 ? args[3] : "illustration")
        default:
            fail("usage: apple-ai status | chat [INSTRUCTIONS] | see IMAGE | imagine PROMPT OUT.png [STYLE]")
        }
    }
}
