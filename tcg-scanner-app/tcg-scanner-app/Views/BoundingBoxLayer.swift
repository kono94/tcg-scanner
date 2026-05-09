//
//  BoundingBoxLayer.swift
//  tcg-scanner-app
//
//  Created by Jan Löwenstrom on 02.02.25.
//  Copyright © 2025 net.lwenstrom. All rights reserved.
//

import Foundation
import SwiftUI

class BoundingBoxLayer: CAShapeLayer {
    private let textLayer = CATextLayer()
    
    override init() {
        super.init()
        setup()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }
    
    private func setup() {
        addSublayer(textLayer)
        fillColor = UIColor.clear.cgColor
        lineWidth = 2
        strokeColor = UIColor.green.cgColor
        
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.fontSize = 14
        textLayer.alignmentMode = .center
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.backgroundColor = UIColor.black.withAlphaComponent(0.7).cgColor
    }
    
    func update(frame: CGRect, label: String) {
        path = UIBezierPath(rect: frame).cgPath
        strokeColor = UIColor.green.cgColor
        
        textLayer.string = label
        textLayer.frame = CGRect(
            x: frame.origin.x,
            y: frame.origin.y - 20,
            width: frame.width,
            height: 20
        )
    }
}
